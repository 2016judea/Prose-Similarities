"""
ALJ 09/03/2019 -> 

Project Gutenberg: https://www.gutenberg.org/


Library Documentation: 
    TextBlob = https://textblob.readthedocs.io/en/dev/

    POS Tags:
    CC coordinating conjunction
    CD cardinal digit
    DT determiner
    EX existential there (like: “there is” … think of it like “there exists”)
    FW foreign word
    IN preposition/subordinating conjunction
    JJ adjective ‘big’
    JJR adjective, comparative ‘bigger’
    JJS adjective, superlative ‘biggest’
    LS list marker 1)
    MD modal could, will
    NN noun, singular ‘desk’
    NNS noun plural ‘desks’
    NNP proper noun, singular ‘Harrison’
    NNPS proper noun, plural ‘Americans’
    PDT predeterminer ‘all the kids’
    POS possessive ending parent’s
    PRP personal pronoun I, he, she
    PRP$ possessive pronoun my, his, hers
    RB adverb very, silently,
    RBR adverb, comparative better
    RBS adverb, superlative best
    RP particle give up
    TO, to go ‘to’ the store.
    UH interjection, errrrrrrrm
    VB verb, base form take
    VBD verb, past tense took
    VBG verb, gerund/present participle taking
    VBN verb, past participle taken
    VBP verb, sing. present, non-3d take
    VBZ verb, 3rd person sing. present takes
    WDT wh-determiner which
    WP wh-pronoun who, what
    WP$ possessive wh-pronoun whose
    WRB wh-abverb where, when
"""

from textblob import TextBlob
import os
import collections
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def main():
    #dict to track words and frequency count
    word_cnt = {}
    #dict to track novel and most used word list objects
    novel_most_used = {}
    #generate an undirected graph using the networkx library
    G = nx.Graph()

    for filename in os.listdir(os.path.join(os.getcwd(), 'novels')):
        print("Starting process for: " + filename)
        try:
            #read in the novel specified as a string
            f = open(os.path.join(os.getcwd(), 'novels', filename), 'r')
            novel = f.read()
            f.close()

            #utilize TextBlob for NLP
            blob = TextBlob(novel)
            for word, tag in blob.tags:
                #if the word is an adjective
                if tag in ['JJ']:    
                    #get frequency of word and throw into a dict or tuple
                    word_cnt.update({word : blob.word_counts[word]})
            #delete TextBlob object for sake of OS efficiency
            del blob
        
            #take object and sort based on frequency
            sorted_obj = sorted(word_cnt.items(), key=lambda kv: kv[1], reverse=True)
            #throw back into a dict object
            sorted_dict = collections.OrderedDict(sorted_obj)
            #once we throw into list we can subscript the object for top 300
            most_used = list(sorted_dict)[:300]
            #throw the novel and the list object of most used words into dict
            novel_most_used.update({filename : most_used})

            #add node to graph
            G.add_node(filename)

        except UnicodeDecodeError as e:
            print("Cannot decode text from: " + filename)
            print("Skipping this novel due to error\n")

    #get a list of all the combinations of novels, these will serve as the graph edges
    graph_edges = combinations(list(novel_most_used.keys()), 2)
    
    for edge in graph_edges:
        set_1 = set(novel_most_used[edge[0]])
        set_2 = set(novel_most_used[edge[1]])

        overlap = set_1 & set_2
        #we take the number of matches between the sets and divide by the length of the set
        match_rate = float(len(overlap)) / len(set_1)
        #nodes that have higher similarity should reflect less of an edge weight (less cost to get to)
        edge_weight = round(1 - match_rate, 2)

        #add edge to graph
        G.add_edge(str(edge[0]), str(edge[1]), weight=edge_weight)

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G,'weight')
    node_labels = nx.get_node_attributes(G, 'name')
    #draw edges
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, nodecolor='r', edge_color='b')
    #draw nodes
    nx.draw_networkx_nodes(G,pos,node_size=3500)
    nx.draw_networkx_labels(G, pos=pos, node_labels=node_labels, nodecolor='r', edge_color='b', font_size=6)

    #run shortest path algorithm
    for source_node in G.nodes():
        for dest_node in G.nodes():
            if dest_node != source_node:
                print("Shortest path from " + source_node + " to " + dest_node + ": " + str(nx.dijkstra_path(G, source_node, dest_node)))
                print("Shortest path length = " + str(nx.shortest_path_length(G, source_node, dest_node)) + '\n')

    plt.show()

if __name__ == "__main__":
    main()

