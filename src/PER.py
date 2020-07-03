#-*-coding: UTF-8 -*-

'''Docstrings for PER computation and phoneme grouping.

This module has two tasks: 1. group reference phonemes and hypothesized phonemes into classes they belonging 
to; 2. for every class, compute its PER and list deleted phonemes, inserted phonemes and confusion phoneme pairs.
                           
'''

import  time

def propre_reader(f_path):
    '''Read data requred for PER computation
    
    Parameters
    ----------
    f_path : str        
        the path of of file '39phn_preprocessing'
        
    Returns
    -------
    prf_ref_list : list
        a list of reference phonemes
    prf_hyp_list : list
        a list of hypothesized phonemes
    prf_conf_list : list
        a list of confidence phonemes
    '''
    with open(f_path, 'r', encoding='utf-8') as f:
        prf_ref_list, prf_hyp_list, prf_conf_list = [], [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            prf_ref_list.append(line[1])
            prf_hyp_list.append(line[2])
            prf_conf_list.append(line[3])

    f.close()
    return prf_ref_list, prf_hyp_list, prf_conf_list

def confidence_cal(class_phoneme_dic, prf_hyp_list, prf_conf_list):   
    '''Compute average confidence for all phonemic classes
    
    Parameters
    ----------
    class_phoneme_dic : dic
        a dictionary of phonemic classes with corersponding phonemes according to judgement from Reynolds and Antonious
    prf_hyp_list : list
        a list of hypothesized phonemes
    prf_conf_list : list
        a list of confidence phonemes
        
    Returns
    -------
    conf_dic : dic
        a dictionary of phonemic classes with corersponding average confidence
    '''
     
    labels = list(class_phoneme_dic.keys())
    conf_dic = {ph_class: [] for ph_class in labels}
    for index, phoneme in enumerate(prf_hyp_list):
        for ph_class, ph_list in class_phoneme_dic.items():
            if phoneme in ph_list:      
                conf_dic[ph_class].append(float(prf_conf_list[index]))
                
    for ph_class, conf_list in conf_dic.items():
        conf_dic[ph_class] = 100 * sum(conf_list)/len(conf_list)
    
    return conf_dic

def PER_cal(prf_ref_list, prf_hyp_list):
    '''Compute the PER 
    
    Parameters
    ----------
    prf_ref_list : list
        a list of reference phonemes
    prf_hyp_list : list
        a list of hypothesized phonemes
        
    Returns
    -------
    per : float
        PER based on given ref and hyp
    correct : list
        a list of coreect phonemes
    deletion : list
        a list of deleted phonemes
    insertion : list
        a list of inserted phonemes
    substitution : ist
        a list of confusion phoneme pairs
    '''
    
    correct, deletion, insertion, substitution = [], [], [], []
    for index, per_ref in enumerate(prf_ref_list):
        per_hyp = prf_hyp_list[index]
        if per_ref != per_hyp:
            if '*' in per_ref:
                insertion.append(per_hyp)
            elif '*' in per_hyp:
                deletion.append(per_ref)
            else:
                substitution.append((per_ref, per_hyp))
        else:
            correct.append((per_ref, per_hyp))
    
    per = (len(substitution) + len(deletion) + len(insertion)) / (len(substitution) + len(deletion) + len(correct))
    return per, correct, deletion, insertion, substitution

def PER_per_class(prf_ref_list, prf_hyp_list, class_phoneme_dic):
    '''Compute the PER for phonemic classes
    
    Parameters
    ----------
    prf_ref_list : list
        a list of reference phonemes
    prf_hyp_list : list
        a list of hypothesized phonemes
    class_phoneme_dic : dic
        a dictionary of phonemic classes with corersponding phonemes
        
    Returns
    -------
    per_class_phoneme_dic : dic
        a dictionary of phonemic classes with corresponding reference and hypothesized phonemes
    phoneme_class_PER : dic
        a dictionary of phonemic classes with corresponding PER
    phoneme_class_Cor : dic
        a dictionary of phonemic classes with a list of correctly recognized phonemes
    phoneme_class_Del : dic
        a dictionary of phonemic classes with a list of corresponding deleted phonemes
    phoneme_class_Ins : dic
        a dictionary of phonemic classes with a list of corresponding inserted phonemes
    phoneme_class_Sub : dic
        a dictionary of phonemic classes with a list of corresponding confusion phoneme pairs
    '''
    per_class_phoneme_dic = {}
    for index, per_ref in enumerate(prf_ref_list):       
        for phonemic_class in class_phoneme_dic.keys():
            if phonemic_class not in per_class_phoneme_dic.keys():
                per_class_phoneme_dic[phonemic_class] = ([], [])
#            if the ref_phoneme is in this class
            if per_ref in class_phoneme_dic[phonemic_class]:                
                per_class_phoneme_dic[phonemic_class][0].append(per_ref)
                per_class_phoneme_dic[phonemic_class][1].append(prf_hyp_list[index])
                continue
#            deletion happens
            elif '*' in per_ref:
#            if the hyp_phoneme is in this class
                if prf_hyp_list[index] in class_phoneme_dic[phonemic_class]:                    
                    per_class_phoneme_dic[phonemic_class][0].append(per_ref)
                    per_class_phoneme_dic[phonemic_class][1].append(prf_hyp_list[index])
            
    phoneme_class_PER, phoneme_class_Cor, phoneme_class_Del, phoneme_class_Ins, phoneme_class_Sub = {}, {}, {}, {}, {}
    for phonemic_class in per_class_phoneme_dic.keys():
        per_class_ref_list, per_class_hyp_list = per_class_phoneme_dic[phonemic_class]
        per, correct, deletion, insertion, substitution = PER_cal(per_class_ref_list, per_class_hyp_list)
        phoneme_class_PER[phonemic_class] = per
        phoneme_class_Cor[phonemic_class] = correct
        phoneme_class_Del[phonemic_class] = deletion
        phoneme_class_Ins[phonemic_class] = insertion
        phoneme_class_Sub[phonemic_class] = substitution

    return per_class_phoneme_dic, phoneme_class_PER, phoneme_class_Cor, phoneme_class_Del, phoneme_class_Ins, phoneme_class_Sub

def PER(scoring_directory, class_phoneme_dic):
    '''Main function to compute PER and group deleted phonemes, inserted phonemes, confusion phoneme 
        pairs in each classes respectively.
    
    Parameters
    ----------
    scoring_directory : str 
        the directory of raw data
    class_phoneme_dic : dic
        a dictionary of phonemic classes with corersponding phonemes according to judgement from Reynolds and Antonious

    Returns
    -------
    phoneme_class_info : tuple        
        a tuple contains information about PER, deletion list, insertion list and substitution list per phonemic class
    conf_dic : dic
        a dictionary of phonemic classes with corersponding average confidence
    '''
    
    print('calculate PER...')
    
    file_path = scoring_directory + 'preprocessed_data/39phn_preprocessing'
    prf_ref_list, prf_hyp_list, prf_conf_list = propre_reader(file_path)

    ''' PER calculation (per phonemic class) '''
    # time2 = time.clock()
    per_class_phoneme_dic, phoneme_class_PER, phoneme_class_Cor, phoneme_class_Del, phoneme_class_Ins, phoneme_class_Sub = PER_per_class(prf_ref_list,
                                                                                               prf_hyp_list, class_phoneme_dic)
    
    ''' Average Confidence calculation (per phonemic class) '''
    conf_dic = confidence_cal(class_phoneme_dic, prf_hyp_list, prf_conf_list)
    
    # print('2 PER calculation (per phonemic class):', time.clock() - time2)
    phoneme_class_info = (per_class_phoneme_dic, phoneme_class_PER, phoneme_class_Cor, phoneme_class_Del, phoneme_class_Ins, phoneme_class_Sub)

    return phoneme_class_info, conf_dic


# if __name__ == '__main__':
#     model_names = ['decode_TIMIT_test_out_dnn2']
#
#     test_file = '../data/'
#     filename = '/39phn_preprocessing'
#     for model_name in model_names:
#         file_path = test_file + model_name + filename
#
#         ''' PER calculation (overall) '''
#         prf_ref_list, prf_hyp_list = propre_reader(file_path)
#         _, _, _, _, _ = PER_cal(prf_ref_list, prf_hyp_list)
#         ''' PER calculation (per phonemic class) '''
#         class_phoneme_dic = {'Plosives': ['b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch'],
#                              'Fricatives': ['s', 'sh', 'z', 'f', 'th', 'v', 'dh', 'h'],
#                              'Nasals': ['m', 'n', 'ng'],
#                              'Semi-vowels': ['l', 'r', 'er', 'w', 'y'],
#                              'Vowels': ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'uh', 'uw'],
#                              'Diphthongs': ['ey', 'aw', 'ay', 'oy', 'ow'],
#                              'Closures': ['sil', 'dx']}
#         PER_per_class(prf_ref_list, prf_hyp_list, class_phoneme_dic)
