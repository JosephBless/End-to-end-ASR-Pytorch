import sys
import re
import csv
import editdistance as ed

# Error rate functions
def cal_cer(hyp, truth):
    return 100*float(ed.eval(hyp, truth))/len(truth)

def cal_wer(hyp, truth, SEP=' '):
    return 100*float(ed.eval(hyp.split(SEP), truth.split(SEP)))/len(truth.split(SEP))

def get_testcases(fname):

    def clean(ref):
        ref = re.sub(r'B\-(\S+) ', '', ref)
        ref = re.sub(r' E\-(\S+)', '', ref)
        return ref

    gex = re.compile(r'B\-(\S+) (.+?) E\-\1')
    c = list(csv.reader(open(fname), delimiter='\t'))
    testcases = []
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

if __name__ == '__main__':
    utterance_files = sys.argv[1]
    test_cases = get_testcases(utterance_files)

    asr_wer = 0.0
    asr_cer = 0.0
    sf_f1 = 0.0
    sf_wer = 0.0
    sf_cer = 0.0
    total_sent = 0
    total_slot = 0
    for test_case in test_cases:
        ref_text  = test_case[0]
        hyp_text  = test_case[1]
        # ASR WER/CER evaluation
        asr_wer += cal_wer(hyp_text, ref_text)
        asr_cer += cal_cer(hyp_text, ref_text)
        # Extract Slots
        ref_slots = test_case[2].split(';')
        hyp_slots = test_case[3].split(';')
        unique_slots = []
        ref_dict = {}
        hyp_dict = {}
        if ref_slots[0] != '':
            for ref_slot in ref_slots:
                v, k = ref_slot.split(':')
                ref_dict.setdefault(k, [])
                ref_dict[k].append(v)
        if hyp_slots[0] != '':
            for hyp_slot in hyp_slots:
                v, k = hyp_slot.split(':')
                hyp_dict.setdefault(k, [])
                hyp_dict[k].append(v)
        # Slot Type F1 evaluation
        if len(hyp_dict.keys()) == 0 and len(ref_dict.keys()) == 0:
            F1 = 1.0
        elif len(hyp_dict.keys()) == 0:
            F1 = 0.0
        elif len(ref_dict.keys()) == 0:
            F1 = 0.0
        else:
            P, R = 0.0, 0.0
            for slot in ref_dict:
                if slot in hyp_dict:
                    R += 1
            R = R / len(ref_dict.keys())
            for slot in hyp_dict:
                if slot in ref_dict:
                    P += 1
            P = P / len(hyp_dict.keys())
            F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
        sf_f1 += F1
        total_sent += 1

        # Slot Value WER/CER evaluation
        unique_slots = list(ref_dict.keys())
        for slot in unique_slots:
            for ref_i, ref_v in enumerate(ref_dict[slot]):
                if slot not in hyp_dict:
                    hyp_v = ''
                    wer = cal_wer(hyp_v, ref_v)
                    cer = cal_cer(hyp_v, ref_v)
                else:
                    min_wer = 100
                    min_cer = 100
                    for hyp_v in hyp_dict[slot]:
                        tmp_wer = cal_wer(hyp_v, ref_v)
                        tmp_cer = cal_cer(hyp_v, ref_v)
                        if min_wer > tmp_wer:
                            min_wer = tmp_wer
                        if min_cer > tmp_cer:
                            min_cer = tmp_cer
                    wer = min_wer
                    cer = min_cer
                sf_wer += wer
                sf_cer += cer
                total_slot += 1

    print('ASR WER:', asr_wer/total_sent)
    print('ASR CER:', asr_cer/total_sent)
    print('Slot Type F1:', sf_f1/total_sent)
    print('Slot Value WER:', sf_wer/total_slot)
    print('Slot Value CER:', sf_cer/total_slot)
