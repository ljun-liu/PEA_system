#-*-coding: UTF-8 -*-
''' Docstrings for Main file.

This module is the entrance of this system. Functions of pre-processing, PER computing and analysis are called here.

Attributes
----------
parser : argparse.ArgumentParser
    A parser of arguments input in the commond scripts. The arguments contain 'model_directory' and 'sys_filename',
    which are directories of models to be analyzed and the name of file for ASR systems' comparison respectively.
args : argparse.Namespace        
    the arguments required in this system
'''
import time
import argparse

from src.Prepro import prepro
from src.PER import PER
from src.Analysis import class_phoneme_analysis, systems_analysis, class_error_table_analysis, class_error_graph_analysis
from src.Analysis import class_token_no_graph_analysis, class_error_detailed_graph_analysis, class_confidence_graph_analysis


parser = argparse.ArgumentParser(description='PEA system for phonemic analysis of ASR systems')
#parser.add_argument('--scoring_directory', type=str, default=None, required=True, 
#                    help='the directories of models to be evaluated (connect directories with \';\')')
parser.add_argument('--scoring_dir', type=str, default=None, required=True, 
                    help='the directory of raw data')
parser.add_argument('--results_dir', type=str, default=None, required=True, 
                    help='the directory of analysis results')
args = parser.parse_args()

def args_extractor(args):    
    '''Extract values of input arguments
    
    Parameters
    ----------
    args : argparse.Namespace        
        the arguments required in this system
        
    Returns
    -------
    scoring_directory : str 
        the directory of raw data
    results_directory : str
        the directory of analysis results
    '''
#    model_directory = args.scoring_directory.split(';')
    scoring_directory = args.scoring_dir
    results_directory = args.results_dir

    if not scoring_directory.endswith('/'):
        scoring_directory = scoring_directory + '/'
    if not results_directory.endswith('/'):
        results_directory = results_directory + '/'

    return scoring_directory, results_directory

def input_print(scoring_directory, results_directory):      
    '''Print values of input
    
    Parameters
    ----------    
    scoring_directory : str 
        the directory of raw data
    results_directory : str
        the directory of analysis results
        
    Returns
    -------
    None
    '''
    print('Inputs are:')
    print('{0:<20}{1}'.format('scoring directory:', scoring_directory))
    print('{0:<20}{1}'.format('results directory:', results_directory))


if __name__ == '__main__':
    start_time = time.process_time()
    ''' -----------------------------------Variables----------------------------------- ''' 
#    iftest = False
    scoring_directory, results_directory = args_extractor(args)
    class_phoneme_dic = {'Plosives': ['b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch'],
                         'Fricatives': ['s', 'sh', 'z', 'f', 'th', 'v', 'dh', 'h'],
                         'Nasals': ['m', 'n', 'ng'],
                         'Semi-vowels': ['l', 'r', 'er', 'w', 'y'],
                         'Vowels': ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'uh', 'uw'],
                         'Diphthongs': ['ey', 'aw', 'ay', 'oy', 'ow'],
                         'Closures': ['sil', 'dx'],
                         'Voiced': ['b', 'd', 'g', 'z', 'v'],
                         'Unvoiced': ['p', 't', 'k', 's','f'],
                         'Consonants': ['b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch',
                                        's', 'sh', 'z', 'f', 'th', 'v', 'dh', 'h',
                                        'm', 'n', 'ng', 'l', 'r', 'er', 'w', 'y'],
                         'Overall': ['b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch', 's', 'sh', 'z', 'f',
                                     'th', 'v', 'dh', 'h', 'm', 'n', 'ng', 'l', 'r', 'er', 'w', 'y',
                                     'iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'uh', 'uw', 'ey', 'aw', 'ay',
                                     'oy', 'ow', 'sil', 'dx']}
                         
    model_per_dic = {}

#    # correctness test (overall test):
#    test_overall_class_phoneme_dic = {'Overall_test': ['b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch', 's', 'sh', 'z', 'f',
#                                                       'th', 'v', 'dh', 'h', 'm', 'n', 'ng', 'l', 'r', 'er', 'w', 'y',
#                                                       'iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'uh', 'uw', 'ey', 'aw', 'ay',
#                                                       'oy', 'ow', 'sil', 'dx']}
#    if iftest:
#        class_phoneme_dic = test_overall_class_phoneme_dic
        
    print('-' *50)
    input_print(scoring_directory, results_directory)
    
    ''' -----------------------------------Implemetation----------------------------------- '''
    print('\nAnalysis started ...\n'+'-' *50)
    
    ''' pre-processing '''
    # start_prepro_time = time.clock()
    prepro(scoring_directory)
    # print('pre-processing duration:', time.clock() - start_prepro_time)

    ''' PER (with confidence) calculation '''
    # start_per_time = time.clock()
    phoneme_class_info, conf_dic = PER(scoring_directory, class_phoneme_dic)
    # print('PER calculation duration:', time.clock() - start_per_time)

    ''' analysis (figures) '''
#    start_analysis_time = time.clock()
    class_error_dic = class_phoneme_analysis(results_directory, phoneme_class_info)
    class_error_table_analysis(results_directory, class_error_dic, conf_dic)
#    start_analysis_time = time.clock()
    class_error_graph_analysis(results_directory, class_error_dic, ifshownum=True)
#    print('analysis (figures) duration:', time.clock() - start_analysis_time)
#    start_analysis_time = time.clock()
    class_token_no_graph_analysis(results_directory, class_error_dic, ifshownum=True)
    class_error_detailed_graph_analysis(results_directory, class_error_dic)
    class_confidence_graph_analysis(results_directory, conf_dic)
#    print('analysis (figures) duration:', time.clock() - start_analysis_time)

    end_time = time.process_time()
    duration = end_time - start_time
    print('Elapsed time: {:8.6} msec'.format(1000*duration))
    
    print('-' *50 + '\nAnalysis ends.\n')




