import argparse
def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--new_file', default='Pubmed/Pubmed_wiki0.2.txt')
    parser.add_argument('--old_file', default='Pubmed_edgelist.txt')
    parser.add_argument('--rho', default=0.2, help='杰卡德系数的值')
    parser.add_argument('--file_label', default='datalatel/Pubmed_labels.txt')

    # args = parser.parse_args()
    return parser.parse_args()