import xml.etree.ElementTree as ET
import pandas as pd
import os
import glob
import linecache
import argparse
import pdb

root_dir = 'data/raw/download/Texts'  # Root directory path
extension = '**/*.xml'  # xml extension

def parse_bnc_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    i = 1
    for s in root.iter('s'):
        sentence_id = s.attrib.get('n')
        for element in s:
            if element.tag == 'w':
                data.append({
                    'XML_ID': os.path.splitext(os.path.basename(file_path))[0],  
                    'SentenceID': sentence_id,
                    'TokenID': i,
                    'Token': element.text,
                    'POS': element.attrib.get('pos'),
                    'Lemma': element.attrib.get('hw'),
                    'c5': element.attrib.get('c5')
                })
                i += 1
            elif element.tag == 'c': # for stopwords (e.g. punctuations)
                data.append({
                    'XML_ID': os.path.splitext(os.path.basename(file_path))[0],
                    'SentenceID': sentence_id,
                    'TokenID': i,
                    'Token': element.text,
                    'POS': 'STOP',
                    'Lemma': element.text,
                    'c5': element.attrib.get('c5')
                })
                i += 1
    return data

def loop_bnc_xml(root_dir, processed_dir="bnc_processed", extension='**/*.xml'):

    for directory in glob.glob(os.path.join(root_dir, '[A-K]'), recursive=False):  # loop over A-K dir
        print(f'Parsing directory {directory}...')
        dir_data = []
        for file_path in glob.glob(os.path.join(directory, extension), recursive=True):
            print(f'Parsing file {file_path}...')
            dir_data.extend(parse_bnc_xml(file_path))  

        df = pd.DataFrame(dir_data)  
        dir_name = os.path.basename(directory)  
        df.to_csv(f'{processed_dir}/{dir_name}.tsv', sep='\t', index=False)  
        print(f'Saved {processed_dir}/{dir_name}.tsv')

def parse_bnc_xml_metadata(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get the second line to check whether speech or written text
    second_line = linecache.getline(file_path, 2).strip()

    title = root.find('.//titleStmt/title')
    name = root.find('.//titleStmt/respStmt/name')
    extent = root.find('.//extent') # token counts info

    xml_id = [v for k, v in root.attrib.items() if 'id' in k]

    return {
        'xml_id': xml_id[0] if xml_id else None,
        'title': title.text if title is not None else None,
        'name': name.text if name is not None else None,
        'extent': extent.text if extent is not None else None,
        'second_line': second_line 
    }


def loop_bnc_xml_metadata(root_dir, extension='**/*.xml'):
    metadata = []

    for file_path in glob.glob(os.path.join(root_dir, '[A-K]', extension), recursive=True):  
        print(f'Parsing file {file_path}...')
        metadata.append(parse_bnc_xml_metadata(file_path))  

    metadata_df = pd.DataFrame(metadata) 
    metadata_df.to_csv('data/processed/metadata.tsv', sep='\t', index=False)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', action='store_true', help='Parse XML files')
    parser.add_argument('--metadata', action='store_true', help='Parse into metadata')
    args = parser.parse_args()

    if args.xml:
        loop_bnc_xml(root_dir)
    elif args.metadata:
        loop_bnc_xml_metadata(root_dir, extension) 