#!/usr/bin/python
import logging
import re
import sys
import csv
from multiprocessing import Pool
from collections import defaultdict
from utterance_aligner import UtteranceAligner

def get_testcases(fname):

    def clean(ref):
        ref = re.sub(r'B\-(\S+) ', '', ref)
        ref = re.sub(r' E\-(\S+)', '', ref)
        return ref

    gex = re.compile(r'B\-(\S+) (.+?) E\-\1')
    c = list(csv.reader(open(fname), delimiter='\t'))
    testcases = []
    #fw = open(sys.argv[2], 'w')
    for idx, hyp, ref in c[1:]:
        hyp = re.sub(r' +', ' ', hyp)
        ref = re.sub(r' +', ' ', ref)
        hyp_slots = gex.findall(hyp)
        ref_slots = gex.findall(ref)
        if len(hyp_slots)>0:
            hyp_slots = ';'.join([':'.join([clean(x[1]), x[0]]) for x in hyp_slots])
            ref_slots = ';'.join([':'.join([x[1], x[0]]) for x in ref_slots])
        else:
            hyp_slots = ''
            ref_slots = ''
        ref = clean(ref)
        hyp = clean(hyp)
        testcase = [ref, hyp, ref_slots, hyp_slots]
        testcases.append(testcase)
    return testcases

def get_indexs(utterance, span):
    tokens = re.split('\s+', utterance)
    span_tokens = re.split('\s+', span)
    for i in range(len(tokens) - len(span_tokens) + 1):
        match = True
        for j, span_token in enumerate(span_tokens):
            if span_token != tokens[i+j]:
                match = False
                break

        if match:
            return (i, i+len(span_tokens)-1)

    return None


def process_slot_types(string):
    return [re.split('\:', span)[0] for span in re.split('\;', string)]


def get_utterance_tp_fn_fp(aligner, transcription, hypothesis, transcription_slots, hypothesis_slots):
    transcription_slot_span_indexs = [
        get_indexs(transcription, transcription_slot) for transcription_slot in transcription_slots]
    hypothesis_slot_span_indexs = [
        get_indexs(hypothesis, hypothesis_slot) for hypothesis_slot in hypothesis_slots]

    if transcription_slot_span_indexs[0] == None:
        # assume insertion errors
        TP, FN, FP = 0, 0, 0
        alignment_trace = None

        for span in hypothesis_slot_span_indexs:
            FP += span[1] - span[0] + 1

        return TP, FN, FP, alignment_trace


    if hypothesis_slot_span_indexs[0] == None:
        # assume deletion errors
        TP, FN, FP = 0, 0, 0
        alignment_trace = None

        for span in transcription_slot_span_indexs:
            FN += span[1] - span[0] + 1

        return TP, FN, FP, alignment_trace

    transcription_slot_span_indexs = set([
        index for indexs in transcription_slot_span_indexs for index in range(indexs[0], indexs[1]+1)])
    hypothesis_slot_span_indexs = set([
        index for indexs in hypothesis_slot_span_indexs for index in range(indexs[0], indexs[1]+1)])

    _, alignment_trace = aligner.align_utterances(transcription, hypothesis)

    transcription_index, hypothesis_index = 0, 0
    TP, FN, FP = 0, 0, 0

    for alignment_symbol in alignment_trace:
        assert transcription_index <= len(re.split('\s+', transcription))
        assert hypothesis_index <= len(re.split('\s+', hypothesis))

        if alignment_symbol == UtteranceAligner.MATCH_SYM:
            if transcription_index in transcription_slot_span_indexs:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    TP += 1
                else:
                    FN += 1
            else:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    FP += 1
                else:
                    # slot = other
                    pass
            transcription_index += 1
            hypothesis_index += 1

        elif alignment_symbol == UtteranceAligner.DELETION_SYM:
            if transcription_index in transcription_slot_span_indexs:
                FN += 1
            else:
                # slot = other
                pass
            transcription_index += 1

        elif alignment_symbol == UtteranceAligner.INSERTION_SYM:
            if hypothesis_index in hypothesis_slot_span_indexs:
                FP += 1
            else:
                # slot = other
                pass
            hypothesis_index += 1

        elif alignment_symbol == UtteranceAligner.SUBSTITUTION_SYM:
            if transcription_index in transcription_slot_span_indexs:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    FP += 1
                    FN += 1
                else:
                    FN += 1
            else:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    FP += 1
                else:
                    # slot = other
                    pass
            transcription_index += 1
            hypothesis_index += 1

        else:
            raise ValueError

    return TP, FN, FP, alignment_trace


def process_test_case(test_case, aligner):
    #assert 5 <= len(test_case) <= 6
    transcription, hypothesis = test_case[0], test_case[1]
    transcription_slots = process_slot_types(test_case[2])
    hypothesis_slots = process_slot_types(test_case[3])

    TP, FN, FP, alignment_trace = get_utterance_tp_fn_fp(
        aligner, transcription, hypothesis, transcription_slots, hypothesis_slots)

    result = []
    if TP:
        result.append('{}:TP'.format(TP))
    if FN:
        result.append('{}:FN'.format(FN))
    if FP:
        result.append('{}:FP'.format(FP))

    result = ';'.join(result)
    #print(result)
    """
    f1_ground_truth = test_case[4]
    if result != f1_ground_truth:
        for line in test_case[:5]:
            print(line)
        print(result)
        print(alignment_trace)
    """
    return TP, FN, FP


def compute_char_edit_distance(token_1, token_2):
    aligner = UtteranceAligner()

    return aligner.align_utterances(
        ' '.join(list(token_1)), ' '.join(list(token_2)))[0]


def substitution_cost(alignment_matrix, i, j, tokens_1, tokens_2):
    """
    Ignore input and return 1 for all cases (insertion, deletion, substitution)
    """
    edit_distance = compute_char_edit_distance(tokens_1[i-1], tokens_2[j-1])
    #print(tokens_1[i-1], tokens_2[j-1], edit_distance/len(tokens_1[i-1]))

    if edit_distance/len(tokens_1[i-1]) < 0.7 or len(tokens_1[i-1]) < 4:
        return 1.5
    else:
        return 10

    #return 1 + min(edit_distance/len(tokens_1), 1)

def insert_delete_cost(alignment_matrix, i, j, tokens_1, tokens_2):
    edit_distance = compute_char_edit_distance(tokens_1[i-1], tokens_2[j-1])

    if edit_distance/len(tokens_1[i-1]) > 0.4:
        return 1
    else:
        return 10

    #return 1 - min(edit_distance/len(tokens_1), 1) * 0.75


def local_f1(test_case, IBO=False):
    aligner = UtteranceAligner(
        substitution_cost=substitution_cost,
        insertion_cost=insert_delete_cost,
        deletion_cost=insert_delete_cost)

    slot2F1  = {} #defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
    ref_text  = test_case[0]
    hyp_text  = test_case[1]
    ref_slots = test_case[2].split(';')
    hyp_slots = test_case[3].split(';')
    unique_ref_slots = []
    if ref_slots[0] == '':
        return slot2F1
        if hyp_slots[0] != '':
            for slot in hyp_slots:
                hyp_slot = slot.split(':')[1]
                slot2F1.setdefault(hyp_slot, [0,0,0])
                slot2F1[hyp_slot][2] += 1
    for ref_slot in ref_slots:
        if IBO:
            slot_name = ref_slot.split(':')[1]
        else:
            #slot_name = ref_slot.split(':')[1].split('-')[1]
            slot_name = ref_slot.split(':')[1]
        unique_ref_slots.append(slot_name)
    unique_ref_slots = set(unique_ref_slots)
    for ref_slot in unique_ref_slots:
        matched_ref_slots = [x for x in ref_slots if ref_slot in x]
        matched_hyp_slots = [x for x in hyp_slots if ref_slot in x]
        if matched_hyp_slots != []:
            sub_test_case = [ref_text, hyp_text, ';'.join(matched_ref_slots), ';'.join(matched_hyp_slots)]
        else:
            sub_test_case = [ref_text, hyp_text, ';'.join(matched_ref_slots), 'shit']
        local_TPs, local_FNs, local_FPs = process_test_case(sub_test_case, aligner)
        #print('Extended SL F1 = 2*TPs/(2*TPs + FPs + FNs) for \"{}\" is: {}'.
        #      format(hyp_text, 2*local_TPs/(2*local_TPs + local_FPs + local_FNs)))
        slot2F1.setdefault(ref_slot, [0,0,0])
        slot2F1[ref_slot][0]  += local_TPs
        slot2F1[ref_slot][1]  += local_FNs
        slot2F1[ref_slot][2]  += local_FPs
    return slot2F1

def main(IBO):

    test_case, TPs, FNs, FPs = [], 0, 0, 0
    utterance_files = sys.argv[1]
    test_cases = get_testcases(utterance_files)
    #with open(sys.argv[1]) as f:
    #    content = f.readlines()
    #content = [x.strip('\n') for x in content]

    slot2F1  = defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
    noise2F1 = defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
    snr2F1   = defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
    #increment = 5
    #test_cases = []
    #for idx in range(int(len(content)/increment)):
    #    test_case = content[idx*increment:idx*increment+increment]
    #    test_cases.append(test_case)

    slot2F1s = Pool(16).map(local_f1, test_cases)
    #for idx, test_case in enumerate(test_cases):
        #utt_id    = test_case[0] # 8k3-8k3084s_AirConditioner_7_SNRdb_40
        #snr       = int(utt_id.split('_')[-1])
        #noise     = utt_id.split('_')[1]
        #print(noise, snr)
    for term in slot2F1s:
        for ref_slot in term.keys():
            slot2F1[ref_slot][0] += term[ref_slot][0]
            slot2F1[ref_slot][1] += term[ref_slot][1]
            slot2F1[ref_slot][2] += term[ref_slot][2]

    all_TPs, all_FNs, all_FPs = 0, 0, 0
    for ref_slot in slot2F1.keys():
        TPs = slot2F1[ref_slot][0]
        FNs = slot2F1[ref_slot][1]
        FPs = slot2F1[ref_slot][2]
        all_TPs += TPs
        all_FNs += FNs
        all_FPs += FPs

    print('Overall F1!')
    print('Extended SL F1 = 2*TPs/(2*TPs + FPs + FNs) for ALL is {}'.
          format((100.0 * 2*all_TPs/(2*all_TPs + all_FPs + all_FNs))))

if __name__ == '__main__':
    print('extended micro F1 score (slot)')
    main(False)
    #print('extended micro F1 score (BIO-slot)')
    #main(True)
