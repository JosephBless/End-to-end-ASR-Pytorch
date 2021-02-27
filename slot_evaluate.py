import sys
import re
import csv

LOOP_OVER_ALL_SLOT = True

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
        #    import pdb
        #    pdb.set_trace()
    return testcases

test_case, TPs, FNs, FPs = [], 0, 0, 0
utterance_files = sys.argv[1]
test_cases = get_testcases(utterance_files)

slot2F1  = {} #defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
for test_case in test_cases:
    ref_text  = test_case[0]
    hyp_text  = test_case[1]
    ref_slots = test_case[2].split(';')
    hyp_slots = test_case[3].split(';')
    unique_slots = []
    if ref_slots[0] == '':
        continue
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
    unique_slots = list(ref_dict.keys())
    if LOOP_OVER_ALL_SLOT:
        unique_slots += [x for x in hyp_dict if x not in ref_dict]
    for slot in unique_slots:
        TP = 0
        FP = 0
        FN = 0
        if slot not in ref_dict: # this never happens in list(ref_dict.keys())
            for hyp_v in hyp_dict[slot]:
                FP += 1
        else:
            for ref_i, ref_v in enumerate(ref_dict[slot]):
                if slot not in hyp_dict:
                    FN += 1
                else:
                    match = False
                    for hyp_v in hyp_dict[slot]:
                        #if ref_i < len(hyp_dict[slot]):
                        #    hyp_v = hyp_dict[slot][ref_i]
                        if hyp_v == ref_v:
                            match = True
                            break
                    if match:
                        TP += 1
                    else:
                        FN += 1
                        FP += 1
        slot2F1.setdefault(slot, [0,0,0])
        slot2F1[slot][0] += TP
        slot2F1[slot][1] += FN
        slot2F1[slot][2] += FP

all_TPs, all_FNs, all_FPs = 0, 0, 0
for slot in slot2F1.keys():
    TPs = slot2F1[slot][0]
    FNs = slot2F1[slot][1]
    FPs = slot2F1[slot][2]
    all_TPs += TPs
    all_FNs += FNs
    all_FPs += FPs
    #print('Extended SL F1 = 2*TPs/(2*TPs + FPs + FNs) for {} is {}'.
    #      format(ref_slot, (100.0 * 2*TPs/(2*TPs + FPs + FNs))))

print('Overall F1!')
print('Extended SL F1 = 2*TPs/(2*TPs + FPs + FNs) for ALL is {}'.
      format((100.0 * 2*all_TPs/(2*all_TPs + all_FPs + all_FNs))))
