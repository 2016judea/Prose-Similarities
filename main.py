"""
ALJ 03/15/2020

"""
import os
from analysis import build_likeness_graph


def main():
    build_likeness_graph(os.path.join(os.getcwd(), 'novels'), show_graph=True, shortest_path=False)

if __name__ == "__main__":
    main()

