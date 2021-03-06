"""
ALJ 03/15/2020

"""
import os
from analysis import build_likeness_graph, sentiment_analysis, sentiment_by_parts, sentiment_multiple_novels
import matplotlib.pyplot as plt


def main():

    graph = build_likeness_graph(os.path.join(os.getcwd(), 'poetry', 'testing'), show_graph=False, shortest_path=False)

    
    #sentiment_by_parts(os.path.join(os.getcwd(), 'novels', 'ThisSideOfParadise.txt'), scale=4, jump=2)
    #sentiment_by_parts(os.path.join(os.getcwd(), 'novels', 'TheBeautifulAndDamned.txt'), scale=4, jump=2)
    #sentiment_by_parts(os.path.join(os.getcwd(), 'novels', 'TheGreatGatsby.txt'), scale=4, jump=2)
    #sentiment_by_parts(os.path.join(os.getcwd(), 'novels', 'TenderIsTheNight.txt'), scale=4, jump=2)
    
    
    novels_to_compare = [
        os.path.join(os.getcwd(), 'novels', 'fitzgerald', 'ThisSideOfParadise.txt'),
        os.path.join(os.getcwd(), 'novels', 'fitzgerald', 'TheBeautifulAndDamned.txt'),
        os.path.join(os.getcwd(), 'novels', 'fitzgerald', 'TheGreatGatsby.txt'),
        os.path.join(os.getcwd(), 'novels', 'fitzgerald', 'TenderIsTheNight.txt')
    ]

    #sentiment_multiple_novels(novels_to_compare, 6)

if __name__ == "__main__":
    main()

