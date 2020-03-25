"""
ALJ 03/15/2020 -> 

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
import re

def get_cleaned_word(to_fix):
    valid = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
     
    ret = '' 
    for char in to_fix:
        if char.upper() in valid:
            ret += char
    
    return ret

def check_regular_chars(test): 
    valid = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
     
    for char in test:
        if char.upper() not in valid:
            return False
    
    return True

def sentiment_multiple_novels(target_novels, parts):
    """Compare list of novels at certain granularity (parts)
    
    Arguments:
        target_novels {[string]} -- [list of absolute paths to novels]
        parts {[int]} -- [granularity to compare novels at]
    """
    for novel in target_novels:
        sentiment_value, sentiment_position = sentiment_analysis(novel, split=parts, show_graph=False)
        plt.plot(sentiment_position, sentiment_value, label='{novel}'.format(novel=os.path.basename(novel)))
    
    plt.xlabel('Portion of Novel')
    plt.ylabel('Sentiment')
    plt.legend(loc='best')
    plt.show()


def sentiment_by_parts(target_novel, scale, jump):
    """View sentiment of novel split into different portions
    
    Arguments:
        target_novel {[string]} -- [absolute os path to target novel]
        scale {[int]} -- [range of analysis]
        jump {[int]} -- [how much to jump between portion analysis]
    """
    for x in range(2, scale+1, jump):
        sentiment_value, sentiment_position = sentiment_analysis(target_novel, split=x, show_graph=False)
        # rescale the points on graph by factor we are ranging through
        factor = scale / x
        for i in range(1, len(sentiment_position)):
            sentiment_position[i] = factor * sentiment_position[i]
        plt.plot(sentiment_position, sentiment_value, label='{x} parts'.format(x=x))
    
    plt.xlabel('Portion of Novel')
    plt.ylabel('Sentiment')
    plt.legend(loc='best')
    plt.show()


def sentiment_analysis(target_novel, split='paragraphs', show_graph=False):
    """Perform sentiment analysis on some novel
    
    Arguments:
        target_novel {[string]} -- [absolute os path to novel text file]
        split {[string or int]} -- [options: 'paragraphs', some number to times to slice]
    """
    f = open(target_novel, 'r')
    novel_as_string = f.read()
    f.close()

    if not str(split).isnumeric() and split == 'paragraphs':
        segments = novel_as_string.split("\n\n")
    
    elif str(split).isnumeric():
        segments = list()
        portion = int(len(novel_as_string) / split)
        start = 0
        end = portion
        for part in range(1, split+1):
            segments.append(novel_as_string[start:end])
            start = end
            end = end + portion
    
    else:
        print('You did not enter a valid sentiment slicing action')
        return
    
    sentiment_value = []
    sentiment_position = []
    x = 0

    for segment in segments:
        x += 1
        #utilize TextBlob for NLP
        blob = TextBlob(segment)
        sentiment_value.append(blob.sentiment.polarity)
        sentiment_position.append(x)
        #delete TextBlob object for sake of OS efficiency
        del blob

    if show_graph:
        plt.plot(sentiment_position, sentiment_value)
        plt.xlabel('Portion of Novel')
        plt.ylabel('Sentiment')
        plt.show()

    return(sentiment_value, sentiment_position)


def build_likeness_graph(novels_dir, show_graph=False, shortest_path=False):
    #dict to track words and frequency count
    word_cnt = {}
    #dict to track novel and most used word list objects
    novel_most_used = {}
    # dict to track total word count per novel
    novel_word_cnt = {}
    #generate an undirected graph using the networkx library
    G = nx.Graph()

    for filename in os.listdir(novels_dir):
        print("Starting process for: " + filename)
        try:
            #read in the novel specified as a string
            f = open(os.path.join(novels_dir, filename), 'r', errors='ignore')
            novel = f.read()
            f.close()
            total_word_cnt = 0
            relevant_pos_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'MD', 'PDT', 'PRP$', 'RB', 
                                    'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

            #utilize TextBlob for NLP
            blob = TextBlob(novel)
            for word, tag in blob.tags:
                total_word_cnt += 1
                if tag in relevant_pos_tags and check_regular_chars(word):   
                    word_cnt.update({word.upper() : blob.word_counts[word]})
        
            #take object and sort based on frequency
            sorted_obj = sorted(word_cnt.items(), key=lambda kv: kv[1], reverse=True)
            #throw into ordered dict and remove words of minimal frequency
            sorted_dict = collections.OrderedDict(sorted_obj)
            sorted_dict = {x:y for x,y in sorted_dict.items() if y!=0}
            # general most used, without frequency
            novel_most_used.update({filename: sorted_dict})
            # word count for novels
            novel_word_cnt.update({filename: total_word_cnt})

            del blob
            del word_cnt
            word_cnt = {}

            #add node to graph
            G.add_node(filename)

        except UnicodeDecodeError as e:
            print("Cannot decode text from: " + filename)
            print("Skipping this novel due to error\n")
    
    #get a list of all the combinations of novels, these will serve as the graph edges
    graph_edges = combinations(list(novel_most_used.keys()), 2)

    
    # find all the words/identifiers that are shared among all the novels
    def remove_all_overlaps(graph_edges, novel_most_used):
        all_overlap = None
        for edge in graph_edges:
            set_1 = set(novel_most_used[edge[0]].keys())
            set_2 = set(novel_most_used[edge[1]].keys())
            curr_overlap = set_1 & set_2

            if all_overlap is None:
                all_overlap = curr_overlap
            else:
                all_overlap = all_overlap & curr_overlap

        # remove the common overlap from all novels from each 
        for novel in novel_most_used:
            for i in list(all_overlap): 
                try: 
                    novel_most_used[novel].pop(i)
                except ValueError: 
                    pass


    # limit most used words per novel
    def limit_number_words(novel_most_used, N=None, cap_at_max=False, display_words=False):
        if cap_at_max:
            max_across = float("inf")
            for novel in novel_most_used:
                max_across = min(max_across, len(novel_most_used[novel]))
        
        for novel in novel_most_used:
            if cap_at_max:
                novel_most_used[novel] = dict(list(novel_most_used[novel].items())[0: max_across])
            else:
                novel_most_used[novel] = dict(list(novel_most_used[novel].items())[0: N])
            
            if display_words:
                print('Novel: ' + str(novel) + str(novel_most_used[novel]))

    
    def simple_overlap(most_used_1, most_used_2, tolerance):
        set_1 = set(most_used_1.keys())
        set_2 = set(most_used_2.keys())
        
        overlap = set_1 & set_2
        #we take the number of matches between the sets and divide by the average length of the sets
        match_rate = float(len(overlap)) / (len(set_1) + len(set_2) / 2)
        edge_weight = round(match_rate, tolerance)
        
        return edge_weight  #larger this number, the more similar

    
    def linear_weight_overlap(most_used_1, most_used_2, word_cnt_1, word_cnt_2, tolerance):
        set_1 = set(most_used_1.keys())
        set_2 = set(most_used_2.keys())
        
        overlap = set_1 & set_2
        overlap_weight = 0
        matches = dict()

        for match in overlap:
            freq_1 = most_used_1[match] / word_cnt_1
            freq_2 = most_used_2[match] / word_cnt_2
            weight = ((10**-10) / abs(freq_1 - freq_2))
            overlap_weight += weight
            matches.update({match: weight})

        edge_weight = round(overlap_weight, tolerance)
        matches = sorted(matches.items(), key=lambda kv: kv[1], reverse=True)[0: 10]
        top_matches = list()
        for word, weight in matches:
            top_matches.append(word)
        
        return edge_weight, top_matches  #larger the edge weight, the more similar


    def exponential_weight_overlap(most_used_1, most_used_2, word_cnt_1, word_cnt_2, tolerance, factor):
        set_1 = set(most_used_1.keys())
        set_2 = set(most_used_2.keys())
        
        overlap = set_1 & set_2
        overlap_weight = 0
        matches = dict()

        for match in overlap:
            freq_1 = most_used_1[match] / word_cnt_1
            freq_2 = most_used_2[match] / word_cnt_2
            weight = ((10**-10) / (abs(freq_1 - freq_2)**factor))
            overlap_weight += weight
            matches.update({match: weight})

        edge_weight = round(overlap_weight, tolerance)
        matches = sorted(matches.items(), key=lambda kv: kv[1], reverse=True)[0: 10]
        top_matches = list()
        for word, weight in matches:
            top_matches.append(word)
        
        return edge_weight, top_matches  #larger the edge weight, the more similar


    remove_all_overlaps(graph_edges, novel_most_used)
    limit_number_words(novel_most_used, N=500, cap_at_max=True, display_words=False)

    # have to re-init this object this combinations returns an iterator
    graph_edges = combinations(list(novel_most_used.keys()), 2)
    print("\nNovels & Similarity Weights:\n")
    
    for edge in graph_edges:
        most_used_1 = novel_most_used[edge[0]]
        most_used_2 = novel_most_used[edge[1]]
        word_cnt_1 = novel_word_cnt[edge[0]]
        word_cnt_2 = novel_word_cnt[edge[1]]
        
        #edge_weight = simple_overlap(most_used_1, most_used_2, 3)
        #edge_weight, top_matches = linear_weight_overlap(most_used_1, most_used_2, word_cnt_1, word_cnt_2, tolerance=5)
        edge_weight, top_matches = exponential_weight_overlap(most_used_1, most_used_2, word_cnt_1, word_cnt_2, tolerance=3, factor=2)

        #add edge to graph & print results
        G.add_edge(str(edge[0]), str(edge[1]), weight=edge_weight)
        print(str(edge[0]) + ' <--> ' + str(edge[1]) +  ' = ' + str(edge_weight))
        if top_matches:
            print('    Top Matches: ' + str(top_matches))
        print('\n')

    if show_graph:
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G,'weight')
        node_labels = nx.get_node_attributes(G, 'name')
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, nodecolor='r', edge_color='b')
        nx.draw_networkx_nodes(G,pos,node_size=3500)
        nx.draw_networkx_labels(G, pos=pos, node_labels=node_labels, nodecolor='r', edge_color='b', font_size=6)
        plt.show()

    #run shortest path algorithm
    if shortest_path:
        for source_node in G.nodes():
            for dest_node in G.nodes():
                if dest_node != source_node:
                    print("Shortest path from " + source_node + " to " + dest_node + ": " + str(nx.dijkstra_path(G, source_node, dest_node)))
                    print("Shortest path length = " + str(nx.shortest_path_length(G, source_node, dest_node)) + '\n')

    return G
