#-*-coding: UTF-8 -*-
'''Docstrings for Preprocessing.

This module aims to pre-process the raw data from the given scoring directory. 
This is implemented by two steps: 1. extracting available results from 'ctm_39phn.filt.prf' 
into a new file named 'ctm_39phn.filt.prfnew'; 2. write data from '.prfnew' in a structured 
format file '39phn_preprocessing'.

'''

import time
import os

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
        
def prfnew_create(new_file_dir, read_path):
    '''Extract available data from '.prf' file into '.prfnew' file

    Parameters
    ----------
    new_file_dir : str
        the directory of the new file '.prfnew'
    read_path: str
        the path of '.prf' file

    Returns
    -------
    None
    '''
    with open(read_path, 'r', encoding='utf-8') as f:
        new_file = []
        header_list = ['id', 'Labels', 'File', 'Channel', 'REF', 'HYP', 'CONF']
        for line in f.readlines():
            line_tmp = line.strip().split(': ')
            header = line_tmp[0]
            if header in header_list:
                new_file.append(line)
    f.close()

    with open(new_file_dir+'ctm_39phn.filt.prfnew', 'w', encoding='utf-8') as f:
        for line in new_file:
            f.writelines(line)
    f.close()

def prfnew_reader(f_path):
    '''Read data from '.prfnew' file

    Parameters
    ----------
    f_path : str
        the path of '.prfnew' file

    Returns
    -------
    id_list : list
        the list of utterance ids
    ref_list : list
        the list of reference phonemes  
    hyp_list : list
        the list of hypothesized phonemes
    labels_list : list
        the list of utterance labels 
    file_list : list
        the list of utterance files 
    channel_list : list
        the list of utterance channels 
    '''
    with open(f_path, 'r', encoding='utf-8') as f:
        id_list, ref_list, hyp_list, conf_list, labels_list, file_list, channel_list = [], [], [], [], [], [], []
        for line in f.readlines():
            line = line.lower().strip().split(': ')
            header = line[0]
            if header not in ['ref', 'hyp', 'conf']:
                eval(header+'_list').append(line[1])
                flag = True
            else:
                content = formatted_list(line[1], header)
                eval(header+'_list').extend(content)
                if flag:
                    flag = False
                    id_list.extend([id_list[-1]] * (len(content)-1))
                    labels_list.extend([labels_list[-1]] * (len(content)-1))
                    file_list.extend([file_list[-1]] * (len(content)-1))
                    channel_list.extend([channel_list[-1]] * (len(content)-1))
                
            def formatted_list(content, header):
                '''
                1. in 'hyp', there are some phonemes 'sil' in 'ref' that are not recognized by the ASR system;
                2. silence in 'ref' is labelled as '(sil)', while in 'hyp' is marked as 'sil';     
                3. the phoneme 'hh' from the raw results is labelled as 'h'.
                therefore, correct these data in the same format
                '''
                length = 8
                content = [content[i:i+length].strip() for i in range(0, len(content), length)]
                if header == 'hyp':
                    content = [element if element != '' else 'sil' for element in content]
                if header == 'ref':
                    content = [element if element != '(sil)' else 'sil' for element in content]  
                
                content = [element if element != 'hh' else 'h' for element in content]
                return content

    f.close()
    return id_list, ref_list, hyp_list, conf_list, labels_list, file_list, channel_list

def prfnew_writer(f_path, id_list, ref_list, hyp_list, conf_list, labels_list, file_list, channel_list):
    '''Write the data into the new structured file '39phn_preprocessing' 

    Parameters
    ----------
    f_path : str
        the path of of the new structured file '39phn_preprocessing'
    id_list: list
        the list of utterance ids
    ref_list : list
        the list of reference phonemes  
    hyp_list: list
        the list of hypothesized phonemes
    labels_list: list        
        the list of utterance labels
    file_list: list        
        the list of utterance files
    channel_list: list        
        the list of utterance channels

    Returns
    -------
    None
    '''
    with open(f_path, 'w', encoding='utf-8') as f:
        for index in range(len(id_list)):
            f.writelines(id_list[index] + '\t' + ref_list[index] + '\t' + hyp_list[index] + '\t' +
                         conf_list[index] + '\t' + labels_list[index] + '\t' + file_list[index] + '\t' + 
                         channel_list[index] + '\n')

    f.close()

def prepro(scoring_directory):
    '''Main function to implement pre-processing.
    
    Parameters
    ----------
    scoring_directory : str 
        the directory of raw data

    Returns
    -------
    None
    '''
    print('pre-process raw data...')
    
    file_dir = scoring_directory + 'preprocessed_data/'
    mk_dir(file_dir)
    prf_read_path = scoring_directory + 'ctm_39phn.filt.prf'
    # time1 = time.clock()
    # rewrite '.prf' file for better data processing
    prfnew_create(file_dir, prf_read_path)
    # print('1 prfnew create:', time.clock() - time1)

    new_read_path = file_dir + 'ctm_39phn.filt.prfnew'
    # time2 = time.clock()
    id_list, ref_list, hyp_list, conf_list, labels_list, file_list, channel_list = prfnew_reader(new_read_path)
    # print('2 prfnew reader:', time.clock() - time2)
    write_path = file_dir + '39phn_preprocessing'
    # time3 = time.clock()
    # write the data into a new file '39phn_preprocessing' after pre-processing
    prfnew_writer(write_path, id_list, ref_list, hyp_list, conf_list, labels_list, file_list, channel_list)
    # print('3 prfnew writer:', time.clock() - time3)


# if __name__ == '__main__':
#     model_name = 'decode_TIMIT_test_out_dnn2'
#     file_dir = '../data/' + model_name + '/'
#     prf_read_path = file_dir + 'score_5/ctm_39phn.filt.prf'
#     prfnew_create(file_dir, prf_read_path)
#
#     new_read_path = file_dir + 'ctm_39phn.filt.prfnew'
#     id_list, prf_ref_dic, prf_hyp_dic = prfnew_reader(new_read_path)
#
#     write_path = file_dir + '39phn_preprocessing'
#     prfnew_writer(write_path, id_list, prf_ref_dic, prf_hyp_dic)
