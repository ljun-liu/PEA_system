#-*-coding: UTF-8 -*-

'''Docstrings for ASR systems' analysis.

This module creates error-detail files for all phonemic classes. Besides, a comparison table is created for
analysis among given ASR systems.

'''

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import copy
from sklearn.metrics import confusion_matrix

def mk_dir(f_dir):
    '''Create the directory if it does not exist
    
    Parameters
    ----------
    f_dir : str        
        the directory to be created
        
    Returns
    -------
    None
    '''
    dir_list = f_dir.split('/')
    cur_dir = ''
    for dir_tmp in dir_list:
        cur_dir = cur_dir + dir_tmp + '/'
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

def conclusion_analysis(phoneme_class_info, phonemic_class):
    '''Compute conclusion-related results of a phonemic class
    
    Parameters
    ----------
    phoneme_class_info : tuple        
        a tuple contains information about PER, deletion list, insertion list and substitution list per phonemic class
    phonemic_class : str
        phonemic class to be analyzed
        
    Returns
    -------
    error_concl : tuple
        a tuple contains percentage of errors occuringand the number of errors
    correct_concl : tuple
        a tuple contains percentage of correct phonemes and the number of correct phonemes
    sub_concl : tuple    
        a tuple contains percentage of substituted phonemes and the number of substituted phonemes
    del_concl : tuple
        a tuple contains percentage of deleted phonemes and the number of deleted phonemes 
    ins_concl : tuple
        a tuple contains percentage of inserted phonemes and the number of inserted phonemes 
    acc : float
        accuracy
    count : tuple
        a tuple contains the number of phonemes in ref, the number of phonemes in hyp and the number of phonemes after alignment 
    '''
    per_class_phoneme_dic, _, _, phoneme_class_Del, phoneme_class_Ins, phoneme_class_Sub = phoneme_class_info
    class_phoneme, deletion, insertion, substitution = per_class_phoneme_dic[phonemic_class], \
                                                            phoneme_class_Del[phonemic_class], \
                                                            phoneme_class_Ins[phonemic_class], \
                                                            phoneme_class_Sub[phonemic_class]
    ref_list, hyp_list = per_class_phoneme_dic[phonemic_class]
    aligned_cnt = len(ref_list)

    ref_cnt = len([phoneme for phoneme in ref_list if '*' not in phoneme])
    hyp_cnt = len([phoneme for phoneme in hyp_list if '*' not in phoneme])
    del_cnt, ins_cnt, sub_cnt = len(deletion), len(insertion), len(substitution)
    error_cnt = del_cnt + ins_cnt + sub_cnt
    correct_cnt = aligned_cnt - error_cnt
    percent_error, percent_correct = error_cnt / ref_cnt, correct_cnt / ref_cnt
    percent_sub, percent_del, percent_ins = sub_cnt / ref_cnt, del_cnt / ref_cnt, ins_cnt / ref_cnt
    acc = (correct_cnt - ins_cnt) / ref_cnt

    error_concl = (percent_error, error_cnt)
    correct_concl = (percent_correct, correct_cnt)
    sub_concl = (percent_sub, sub_cnt)
    del_concl = (percent_del, del_cnt)
    ins_concl = (percent_ins, ins_cnt)
    count = (ref_cnt, hyp_cnt, aligned_cnt)

    return error_concl, correct_concl, sub_concl, del_concl, ins_concl, acc, count


def confusion_pairs(substitution):
    '''Sort confusion pairs by count
    
    Parameters
    ----------
    substitution : list        
        a list of phoneme pairs in substitution
        
    Returns
    -------
    confusion_pair_list : list
        a list of sorted confusion pairs
    substitution_count : int
        the occurrence of substitution
    '''
    confusion_pair_list = []
    for element in set(substitution):
        pair_cnt = substitution.count(element)
        per_ref, per_hyp = element
        confusion_pair_list.append((per_ref, per_hyp, pair_cnt * (-1)))
        
    substitution_count = sum([cnt for _, _, cnt in confusion_pair_list]) * (-1)
    confusion_pair_list = sorted(confusion_pair_list, key=lambda x: (x[2], x[0]))

    return confusion_pair_list, substitution_count

def insertion_analysis(insertion):
    '''Sort inserted phoneme by count
    
    Parameters
    ----------
    insertion : list        
        a list of inserted phoneme
        
    Returns
    -------
    insertion_list : list
        a list of sorted inserted phoneme
    insertion_count : int
        the occurrence of insertion
    '''
    insertion_list = []
    for element in set(insertion):
        insertion_cnt = insertion.count(element)
        insertion_list.append((element, insertion_cnt * (-1)))

    insertion_list = sorted(insertion_list, key=lambda x: (x[1], x[0]))
    insertion_count = sum([cnt for _, cnt in insertion_list]) * (-1)
    return insertion_list, insertion_count

def deletion_analysis(deletion):
    '''Sort deleted phoneme by count
    
    Parameters
    ----------
    deletion : list        
        a list of deleted phoneme
        
    Returns
    -------
    deletion_list : list
        a list of sorted deleted phoneme
    deletion_count : int
        the occurrence of deletion
    '''
    deletion_list = []
    for element in set(deletion):
        deletion_cnt = deletion.count(element)
        deletion_list.append((element, deletion_cnt * (-1)))

    deletion_list = sorted(deletion_list, key=lambda x: (x[1], x[0]))
    deletion_count = sum([cnt for _, cnt in deletion_list]) * (-1)
    return deletion_list, deletion_count

# implemented with np.array
# def substituted_analysis(confusion_pair_list):
#     time_start = time.clock()
#     substituted = np.array([(substituted, cnt) for substituted, _, cnt in confusion_pair_list])
#     substituted_list = []
#     # print(set(substituted[:, 0]))
#     for element in set(substituted[:, 0]):
#         substituted_cnt = np.sum(substituted[substituted[:, 0] == element, 1].astype(int))
#         substituted_list.append((element, substituted_cnt))
#
#     substituted_list = sorted(substituted_list, key=lambda x: x[1], reverse=True)
#     print('substituted(numpy) duration', time.clock() - time_start)
#     return substituted_list

# implemented with list
def substituted_analysis(confusion_pair_list):
    '''Sort substituted phoneme by count
    
    Parameters
    ----------
    confusion_pair_list : list     
        a list of phoneme pairs in substitution
        
    Returns
    -------
    substituted_list : list
        a list of sorted substituted phoneme
    '''
    substituted = [(sub, cnt) for sub, _, cnt in confusion_pair_list]
    substituted_list = []
    for element in set([sub for sub, _ in substituted]):
        substituted_cnt = sum([cnt for sub, cnt in substituted if sub == element])
        substituted_list.append((element, substituted_cnt))

    substituted_list = sorted(substituted_list, key=lambda x: (x[1], x[0]))
    return substituted_list

# implemented with np.array
# def substituting_analysis(confusion_pair_list):
#     substituting = np.array([(substituting, cnt) for _, substituting, cnt in confusion_pair_list])
#     substituting_list = []
#     # print(set(substituting[:, 0]))
#     for element in set(substituting[:, 0]):
#         substituted_cnt = np.sum(substituting[substituting[:, 0] == element, 1].astype(int))
#         substituting_list.append((element, substituted_cnt))
#
#     substituting_list = sorted(substituting_list, key=lambda x: x[1], reverse=True)
#     return substituting_list

# implemented with list
def substituting_analysis(confusion_pair_list):    
    '''Sort substituting phoneme by count
    
    Parameters
    ----------
    confusion_pair_list : list     
        a list of phoneme pairs in substitution
        
    Returns
    -------
    substituting_list : list
        a list of sorted substituting phoneme
    '''
    time_start = time.clock()
    substituting = [(sub, cnt) for _, sub, cnt in confusion_pair_list]
    substituting_list = []
    for element in set([sub for sub, _ in substituting]):
        substituted_cnt = sum([cnt for sub, cnt in substituting if sub == element])
        substituting_list.append((element, substituted_cnt))

    substituting_list = sorted(substituting_list, key=lambda x: (x[1], x[0]))
    # print('substituting(list) duration', time.clock() - time_start, '\n')
    return substituting_list

def dtl_create(f_path, phoneme_class_info):      
    '''Create error-detail files '.dtl' for all phonemic classes
    
    Parameters
    ----------
    f_path : str     
        the path of '.dtl' file
    phoneme_class_info : tuple     
        a tuple contains information about PER, deletion list, insertion list and substitution list per phonemic class
        
    Returns
    -------
    class_error_dic : dic
        a dictonary of phonemic classes with errors information (PER/Ins/Del/Sub)
    '''
    _, phoneme_class_PER, _, phoneme_class_Del, phoneme_class_Ins, phoneme_class_Sub = phoneme_class_info
    phonemic_classes = phoneme_class_PER.keys()
    
    if not f_path.endswith('/'):
        f_path = f_path + '/'

    class_error_dic = {}
    
    for phonemic_class in phonemic_classes:
        with open(f_path+'39phn_'+phonemic_class+'.dtl', 'w', encoding='utf-8') as f:
            ''' conclusion '''
            error_concl, correct_concl, sub_concl, del_concl, ins_concl, acc, count = conclusion_analysis(phoneme_class_info,
                                                                                                     phonemic_class)
            class_error_dic[phonemic_class] = (error_concl, correct_concl, sub_concl, del_concl, ins_concl, count)
            # correctness test (overall test):
            # print('phonemoc class:{0} PER:{1:>7.1f}'.format(phonemic_class, 100 * error_concl[0]))
            f.writelines('WORD RECOGNITION PERFORMANCE of {}\n\n'.format(phonemic_class))
            f.writelines('{0:<26}={1:>7.1f}%{a:3}({2:>4})\n\n'.format('Percent Total Error', 100 * error_concl[0],
                                                                      error_concl[1], a=''))
            f.writelines('{0:<26}={1:>7.1f}%{a:3}({2:>4})\n\n'.format('Percent Correct', 100 * correct_concl[0],
                                                                      correct_concl[1], a=''))
            f.writelines('{0:<26}={1:>7.1f}%{a:3}({2:>4})\n'.format('Percent Substitution', 100 * sub_concl[0],
                                                                    sub_concl[1], a=''))
            f.writelines('{0:<26}={1:>7.1f}%{a:3}({2:>4})\n'.format('Percent Deletions', 100 * del_concl[0],
                                                                    del_concl[1], a=''))
            f.writelines('{0:<26}={1:>7.1f}%{a:3}({2:>4})\n'.format('Percent Insertions', 100 * ins_concl[0],
                                                                    ins_concl[1], a=''))
            f.writelines('{0:<26}={1:>7.1f}%{a:3}\n\n\n'.format('Percent Word Accuracy', 100 * acc, a=''))
            f.writelines('{0:<26}={a:11}({1:>4})\n'.format('Ref. words', count[0], a=''))
            f.writelines('{0:<26}={a:11}({1:>4})\n'.format('Hyp. words', count[1], a=''))
            f.writelines('{0:<26}={a:11}({1:>4})\n\n'.format('Aligned words', count[2], a=''))

            ''' details - confusion pairs '''
            confusion_pair_list, substitution_count = confusion_pairs(phoneme_class_Sub[phonemic_class])
            f.writelines('{0:<33}{1:<21} ({2})\n'.format('CONFUSION PAIRS', 'Total', len(confusion_pair_list)))
            f.writelines('{0:<33}{1:<21} ({2})\n\n'.format('', 'With >=  1 occurances', len(confusion_pair_list)))
            for index, element in enumerate(confusion_pair_list):
                per_ref, per_hyp, pair_cnt = element
                f.writelines('{0:>4}:{1:>5}  ->  {2:} ==> {3}\n'.format(index+1, pair_cnt * (-1), per_ref, per_hyp))
            f.writelines('{:>5}-------\n'.format(''))
            f.writelines('{0:>5}{1:5}\n\n\n\n'.format('', substitution_count))

            ''' details - insertion '''
            insertion_list, insertion_count = insertion_analysis(phoneme_class_Ins[phonemic_class])
            f.writelines('{0:<33}{1:<21} ({2})\n'.format('INSERTIONS', 'Total', len(insertion_list)))
            f.writelines('{0:<33}{1:<21} ({2})\n\n'.format('', 'With >=  1 occurances', len(insertion_list)))
            for index, element in enumerate(insertion_list):
                inserted, insertion_cnt = element
                f.writelines('{0:>4}:{1:>5}  ->  {2:}\n'.format(index+1, insertion_cnt * (-1), inserted))
            f.writelines('{:>5}-------\n'.format(''))
            f.writelines('{0:>5}{1:5}\n\n\n\n'.format('', insertion_count))

            ''' details - deletion '''
            deletion_list, deletion_count = deletion_analysis(phoneme_class_Del[phonemic_class])
            f.writelines('{0:<33}{1:<21} ({2})\n'.format('DELETIONS', 'Total', len(deletion_list)))
            f.writelines('{0:<33}{1:<21} ({2})\n\n'.format('', 'With >=  1 occurances', len(deletion_list)))
            for index, element in enumerate(deletion_list):
                deleted, deletion_cnt = element
                f.writelines('{0:>4}:{1:>5}  ->  {2:}\n'.format(index+1, deletion_cnt * (-1), deleted))
            f.writelines('{:>5}-------\n'.format(''))
            f.writelines('{0:>5}{1:5}\n\n\n\n'.format('', deletion_count))

            ''' details - substitution (substituted) '''
            substituted_list = substituted_analysis(confusion_pair_list)
            f.writelines('{0:<33}{1:<21} ({2})\n'.format('SUBSTITUTIONS', 'Total', len(substituted_list)))
            f.writelines('{0:<33}{1:<21} ({2})\n\n'.format('', 'With >=  1 occurances', len(substituted_list)))
            for index, element in enumerate(substituted_list):
                substituted, substitution_cnt = element
                f.writelines('{0:>4}:{1:>5}  ->  {2:}\n'.format(index+1, substitution_cnt * (-1), substituted))
            f.writelines('{:>5}-------\n'.format(''))
            f.writelines('{0:>5}{1:5}\n\n\n'.format('', substitution_count))
            f.writelines('* NOTE: The \'Substitution\' words are those reference words\n'
                         '        for which the recognizer supplied an incorrect word.\n\n\n')

            ''' details - substitution (substituting) '''
            substituting_list = substituting_analysis(confusion_pair_list)
            f.writelines('{0:<33}{1:<21} ({2})\n'.format('FALSELY RECOGNIZED', 'Total', len(substituting_list)))
            f.writelines('{0:<33}{1:<21} ({2})\n\n'.format('', 'With >=  1 occurances', len(substituting_list)))
            for index, element in enumerate(substituting_list):
                substituted, substitution_cnt = element
                f.writelines('{0:>4}:{1:>5}  ->  {2:}\n'.format(index+1, substitution_cnt * (-1), substituted))
            f.writelines('{:>5}-------\n'.format(''))
            f.writelines('{0:>5}{1:5}\n\n\n'.format('', substitution_count))
            f.writelines('* NOTE: The \'Falsely Recognized\' words are those hypothesis words\n'
                         '        which the recognizer incorrectly substituted for a reference word.')

        f.close()
        
    return class_error_dic

def class_phoneme_analysis(results_directory, phoneme_class_info):     
    '''Print detailed analysis for phonemic classes
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    phoneme_class_info : tuple     
        a tuple contains information about PER, deletion list, insertion list and substitution list per phonemic class
        
    Returns
    -------
    class_error_dic : dic
        a dictonary of phonemic classes with errors information (PER/Ins/Del/Sub)
    '''
        
    print('analyze PER for each phonemic class...')
    mk_dir(results_directory)
    class_error_dic = dtl_create(results_directory, phoneme_class_info)
    
    return class_error_dic


def class_error_table_analysis(results_directory, class_error_dic, conf_dic): 
    '''Print table about phonemic classes and errors information (PER/Ins/Del/Sub)
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    class_error_dic : dic
        a dictonary of phonemic classes with errors information (PER/Ins/Del/Sub)
    conf_dic : dic
        a dictionary of phonemic classes with corersponding average confidence
        
    Returns
    -------
    None
    '''
    
    print('draw class_errorinfo table...')
        
    with open(results_directory+'classes_errors.txt', 'w', encoding='utf-8') as f:
        ''' head of table'''
        f.writelines('-' * (16+1) * 7)
        f.writelines('\n|{0:^16}|{1:^16}|{2:^16}|{3:^16}|{4:^16}|{5:^16}|{6:^16}'.format('Classes', 'Deletion(%)', 'Insertion(%)', 
                                                                         'Substitution(%)', 'PER(%)', '# Phonemes', 'Confidence(%)'))
        f.writelines('|\n' + '-' * (16+1) * 7)

        ''' body of table'''
        for phonemic_class, errors_info in class_error_dic.items():
#            print(phonemic_class)
            error_concl, _, sub_concl, del_concl, ins_concl, count = errors_info
            f.writelines('\n|{0:^16}|{1:^16.1f}|{2:^16.1f}|{3:^16.1f}|{4:^16.1f}|{5:^16}|{6:^16.1f}'.format(phonemic_class, 100 * del_concl[0],
                                                                                                     100 * ins_concl[0], 100 * sub_concl[0], 
                                                                                                     100 * error_concl[0], count[0], conf_dic[phonemic_class]))
            f.writelines('|\n' + '-' * (16+1) * 7)

    f.close()

def class_error_graph_analysis(results_directory, class_error_dic, width=0.2, ifshownum=True): 
    '''Print bar graph about phonemic classes and errors information (PER/Ins/Del/Sub)
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    class_error_dic : dic
        a dictonary of phonemic classes with errors information (PER/Ins/Del/Sub)
    width : float
        the width of the bar
    ifshownum : bool
        True for showing numbers; False for not
        
    Returns
    -------
    None
    '''
    
    print('draw class_errorinfo graph...')
    
    deletion, insertion, substitution, PER = [], [], [], []
    for _, errors_info in class_error_dic.items():
        error_concl, _, sub_concl, del_concl, ins_concl, count = errors_info
        deletion.append(round(100 * del_concl[0], 1))
        insertion.append(round(100 * ins_concl[0], 1))
        substitution.append(round(100 * sub_concl[0], 1))
        PER.append(round(100 * error_concl[0], 1))
    
    labels = class_error_dic.keys()
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(12,8))    
    rects1 = ax.bar(x - width*3/2, deletion, width, label='Deletion')
    rects2 = ax.bar(x - width/2, insertion, width, label='Insertion')
    rects3 = ax.bar(x + width/2, substitution, width, label='Substitution')
    rects4 = ax.bar(x + width*3/2, PER, width, label='PER')
    
    ax.set_ylabel('Percentage (%)')    
    ax.set_xlabel('Phonemic Classes')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
    if ifshownum:
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        
    fig.tight_layout()
        
    figure_dir  = results_directory+'/figures/'
    mk_dir(figure_dir)
    plt.savefig(figure_dir+'classes_errors.pdf')
    
    plt.close()
    
def class_token_no_graph_analysis(results_directory, class_error_dic, width=0.6, ifshownum=True): 
    '''Print graph about phonemic classes and #tokens
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    class_error_dic : dic
        a dictonary of phonemic classes with errors information (PER/Ins/Del/Sub)
    width : float
        the width of the bar
    ifshownum : bool
        True for showing numbers; False for not
        
    Returns
    -------
    None
    '''
    
    print('draw class_token_no graph...')
    sorted_class_error = sorted(class_error_dic.items(), key=lambda x:x[1][5][0])
    ref_cnt, labels = [], []
    for ph_class, errors_info in sorted_class_error:
        _, _, _, _, _, count = errors_info
        labels.append(ph_class)
        ref_cnt.append(count[0])
    
    x = np.arange(len(labels))
    colours = ['#1f77b4'] * (len(labels)-1)
    colours.append('#d62728')
    
    fig, ax = plt.subplots(figsize=(12,8))    
    rects = ax.bar(x, ref_cnt, width, color=colours)
    ax.set_ylabel('# Phonemes in REF')    
    ax.set_xlabel('Phonemic Classes')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
        
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
    if ifshownum:
        autolabel(rects)
    fig.tight_layout()
    
    figure_dir  = results_directory+'/figures/'
    mk_dir(figure_dir)
    plt.savefig(figure_dir+'classes_errors_token_no_sorted.pdf')    
    
    plt.close()
    
def class_error_detailed_graph_analysis(results_directory, class_error_dic, width=0.6, ifshownum=True): 
    '''Print detailed bar graph about phonemic classes and errors information (# tokens of PER/Ins/Del/Sub)
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    class_error_dic : dic
        a dictonary of phonemic classes with errors information (PER/Ins/Del/Sub)
    width : float
        the width of the bar
    ifshownum : bool
        True for showing numbers; False for not
        
    Returns
    -------
    None
    '''
    
    print('draw detailed class_errorinfo graph (# tokens)...')
    y, labels = [], []
    index = [0, 2, 3, 4]
    title = ['PER', 'Substitution', 'Deletion', 'Insertion']
    class_error_dic_tmp = copy.deepcopy(class_error_dic)
    overall = class_error_dic_tmp.pop('Overall')
    for i in range(len(index)):
        sorted_class_error = sorted(class_error_dic_tmp.items(), key=lambda x:x[1][index[i]][0])
        label = [ph_class for ph_class, _ in sorted_class_error]
        label.append('Overall')
        labels.append(label)
        y_value = [errors_info[index[i]] for _, errors_info in sorted_class_error]
        y_value.append(overall[index[i]])
        y.append(np.array(y_value))
    
    x = np.arange(len(labels[0]))
    colours = ['#1f77b4'] * (len(labels[0])-1)
    colours.append('#d62728')
    
    def autolabel(rects, i):
        for index, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate('{:.0f}'.format(y[i][index,1]), xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
    fig = plt.figure(figsize=(12, 8))           
    
    for i in range(len(y)):
        ax = fig.add_subplot(eval('22'+str(i+1)))  
        rects = ax.bar(x, 100 * y[i][:,0], width, color=colours)
        ax.set_ylabel('Percentage (%)')    
        ax.set_xlabel('Phonemic Classes')
        ax.set_title(title[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels[i], fontsize=5.4)
        if ifshownum:
            autolabel(rects, i)
        
        
    fig.tight_layout()
    
    figure_dir  = results_directory+'/figures/'
    mk_dir(figure_dir)
    plt.savefig(figure_dir+'detailed_classes_errors_sorted.pdf')    
    
    plt.close()
    
def class_confidence_graph_analysis(results_directory, conf_dic, width=0.6, ifshownum=True):
    '''Print a graph about the average confidence of phonemic classes
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    conf_dic : dic
        a dictionary of phonemic classes with corersponding average confidence
    width : float
        the width of the bar
    ifshownum : bool
        True for showing numbers; False for not
        
    Returns
    -------
    None
    '''
    
    print('draw class_confidence graph...')
    
    labels = conf_dic.keys()
    avg_conf = [conf_dic[ph_class] for ph_class in labels]
            
    x = np.arange(len(labels))
    colours = ['#1f77b4'] * (len(labels)-1)
    colours.append('#d62728')
    
    fig, ax = plt.subplots(figsize=(12,8))    
    rects = ax.bar(x,avg_conf, width, color=colours)
    ax.set_ylabel('Average Confidence (%)')    
    ax.set_xlabel('Phonemic Classes')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
        
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.1f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
    if ifshownum:
        autolabel(rects)
    fig.tight_layout()
    
    figure_dir  = results_directory+'/figures/'
    mk_dir(figure_dir)
    plt.savefig(figure_dir+'classes_avg_confidence.pdf')    
    
    plt.close()
    
def confusion_matrix_statistics(phoneme_class_Cor, phoneme_class_Sub, ph_classes, class_phoneme_dic):
    '''Print detailed analysis for phonemic classes
    
    Parameters
    ----------
    phoneme_class_Cor : dic
        a dictionary of phonemic classes with a list of correctly recognized phonemes
    phoneme_class_Sub : dic
        a dictionary of phonemic classes with a list of corresponding confusion phoneme pairs
    ph_classes : list     
        a list of phonemic classes in confusion matrix
    class_phoneme_dic : dic
        a dictionary of phonemic classes with corersponding phonemes        
        
    Returns
    -------
    cm : dic
        confusion matrix
    '''
    ph_true, ph_pred = [], []
    for ph_class in ph_classes:
        for _ in phoneme_class_Cor[ph_class]:
            ph_true.append(ph_class)
            ph_pred.append(ph_class)
        for _, hyp in phoneme_class_Sub[ph_class]:
            for hyp_class in ph_classes:
                if hyp in class_phoneme_dic[hyp_class]:
                    ph_true.append(ph_class)
                    ph_pred.append(hyp_class)
                    continue
            
    cm = confusion_matrix(ph_true, ph_pred, labels=ph_classes)
    
    return cm

def confusion_matrix_analysis(results_directory, f_name, phoneme_class_info, ph_classes, class_phoneme_dic):   
    '''Draw confusion matric table
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    f_name : str
        the file name
    phoneme_class_info : tuple     
        a tuple contains information about PER, deletion list, insertion list and substitution list per phonemic class
    ph_classes : list     
        a list of phonemic classes in confusion matrix
    class_phoneme_dic : dic
        a dictionary of phonemic classes with corersponding phonemes    
        
    Returns
    -------
    None
    '''
        
    print('draw ' + f_name + ' confusion matrix...')
    
    _, _, phoneme_class_Cor, _, _, phoneme_class_Sub = phoneme_class_info
    cm = confusion_matrix_statistics(phoneme_class_Cor, phoneme_class_Sub, ph_classes, class_phoneme_dic)
    
    with open(results_directory+f_name+'_confusion_matrix.txt', 'w', encoding='utf-8') as f:  
#       confusion matrix
        f.writelines('Confusion Matrix:\n' + '-' * (16+1) * (len(ph_classes)+1) + '\n|{0:16}'.format(''))        
        for i in range(len(ph_classes)):
            f.writelines('|{0:^16}'.format(ph_classes[i]))
        f.writelines('|\n' + '-' * (16+1) * (len(ph_classes)+1))

        for i in range(len(ph_classes)):
            f.writelines('\n|{0:^16}'.format(ph_classes[i]))
            for j in range(len(ph_classes)):
                f.writelines('|{0:^16}'.format(cm[i,j]))
            f.writelines('|\n' + '-' * (16+1) * (len(ph_classes)+1))
            
#        description
        f.writelines('\n(ROWS: Actual Classes; COLUMNS: Predicted Classes)')
            
#       correct phonemes per class
        f.writelines('\n\n# Correct Phonemes:\n------------------'.format(''))   
        for ph_class in ph_classes:
            f.writelines('\n{0:14}{1:4}'.format(ph_class+':', len(phoneme_class_Cor[ph_class])))
    f.close()
    
def five_broad_confusion_matrix_analysis(results_directory, phoneme_class_info):    
    '''Draw five broad confusion matric table
    
    Parameters
    ----------
    results_directory : str
        the directory of analysis results
    phoneme_class_info : tuple     
        a tuple contains information about PER, deletion list, insertion list and substitution list per phonemic class
        
    Returns
    -------
    None
    '''
        
    print('draw five_broad confusion matrix...')
                 
    
    five_broad_class_phoneme_dic = {'Plosives': ['b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch'],
                                     'Fricatives': ['s', 'sh', 'z', 'f', 'th', 'v', 'dh', 'h'],
                                     'Nasals': ['m', 'n', 'ng'],
                                     'Vowels': ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'uh', 'uw', 'l', 'r', 'er', 'w', 'y', 'ey', 'aw', 'ay', 'oy', 'ow'],
                                     'Closures': ['sil', 'dx']}
    ph_classes = list(five_broad_class_phoneme_dic.keys())
    _, _, phoneme_class_Cor, _, _, phoneme_class_Sub = phoneme_class_info
    phoneme_class_Cor['Vowels'] = phoneme_class_Cor['Vowels'] + phoneme_class_Cor['Semi-vowels'] + phoneme_class_Cor['Diphthongs']
    phoneme_class_Sub['Vowels'] = phoneme_class_Sub['Vowels'] + phoneme_class_Sub['Semi-vowels'] + phoneme_class_Sub['Diphthongs']
    cm = confusion_matrix_statistics(phoneme_class_Cor, phoneme_class_Sub, ph_classes, five_broad_class_phoneme_dic)
    
    with open(results_directory+'five_broad_confusion_matrix.txt', 'w', encoding='utf-8') as f:  
#       confusion matrix
        f.writelines('Confusion Matrix:\n' + '-' * (16+1) * (len(ph_classes)+1) + '\n|{0:16}'.format(''))        
        for i in range(len(ph_classes)):
            f.writelines('|{0:^16}'.format(ph_classes[i]))
        f.writelines('|\n' + '-' * (16+1) * (len(ph_classes)+1))

        for i in range(len(ph_classes)):
            f.writelines('\n|{0:^16}'.format(ph_classes[i]))
            for j in range(len(ph_classes)):
                f.writelines('|{0:^16}'.format(cm[i,j]))
            f.writelines('|\n' + '-' * (16+1) * (len(ph_classes)+1))
            
#        description
        f.writelines('\n(ROWS: Actual Classes; COLUMNS: Predicted Classes)')
            
#       correct phonemes per class
        f.writelines('\n\n# Correct Phonemes:\n------------------'.format(''))   
        for ph_class in ph_classes:
            f.writelines('\n{0:14}{1:4}'.format(ph_class+':', len(phoneme_class_Cor[ph_class])))
    f.close()
    
