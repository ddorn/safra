# Master's Semester project

Title: **Between decideable logics: ω-automata and infinite games**

This projects contains two parts:
- [a thesis](./out/projet.pdf), with proofs and explanations about interesting
    topics involving automata on infinite words, monadic second order logic,
    mu-calculus and infinite games of perfect informations on two players. Care has
    been taken to include visuals to ease the understanding.

- [a python script](./safra.py) to compute the Safra construction and visualize
    it. The rest of this document is dedicated to the usage of this script.

## Dependencies

In order to run this script, you need:
- a linux machine (this might run on a Mac, but has not been tested);
- a recent python version (at least 3.8);
- the python packages `click` and `dot2tex`, which can be installed with
    ```sh
    pip install click dot2tex
    ```
- the [graphviz](https://www.graphviz.org/) software. Instruction for installation is [availaible there](https://graphviz.org/download/);
- `pdflatex` which is included in you favorite latex distribution.

## Usage

To use the script, download or clone this repository.
You can the run (inside the downloaded folder) `python safra.py --help` to see all the available options.

```
Usage: safra.py [OPTIONS] [AUTOMATON]

  Draw and convert a Buchi automaton into a Muller automaton.

  To specifiy which automaton to use, modify the AUTOMATA list in the code,
  and then set the AUTOMATON argument to the index of the automaton in the
  list.

Options:
  -n, --no-trees               Don't draw the trees inside the nodes of the
                               Muller automaton.
  -s, --safra-transition TEXT  Draw the steps to build a specific transition
                               in the Muller automata. Pass a word of 0 and 1
                               and it will show the last transition when
                               reading this word.
  --prog TEXT                  The program to use to render the graph. "dot"
                               and "neato" work best.
  --edge-len FLOAT             Target length for edges for the neato layout.
                               [default=1.0]
  -o, --output TEXT            Latex file to store the graph to.
                               [default=graph.tex]
  -b, --draw-buchi             Draw the Buchi automaton instead of the Muller
                               automton.
  --help                       Show this message and exit.
```