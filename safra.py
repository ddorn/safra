#!/bin/env python3

"""
Conversion from non-deterministic Büchi automata to deterministic Muller
automata via the Safra construction.

This script includes visualisation of the automata with the help of Graphviz and Tikz/Latex.
Both must be installed to use this script.

Dependencies:
 - graphviz: https://graphviz.org/
 - python: pip install dot2tex click
 - pdflatex, from you favorite latex distribution.




Author: Diego Dorn, 2022
License: WTFPL
"""

from __future__ import annotations

import dataclasses
import subprocess
from collections import deque
from dataclasses import dataclass
from pprint import pprint
from random import randrange, shuffle
from textwrap import indent
from typing import Generator, Iterator

import click
import dot2tex

fset = frozenset
Label = str | int
Transition = dict[tuple[Label, int], set[Label]]
DetTransition = dict[tuple[Label, int], Label]


@dataclass(frozen=True)
class SafraNode:
    """
    A node in the Safra tree.
    """
    name: int
    label: fset[Label]
    children: tuple[int, ...]
    marked: bool = False


class SafraTree(dict[int, SafraNode]):
    """
    A Safra tree, where each node is labeled with an integer.
    """

    @classmethod
    def new(cls, initial_state: Label) -> SafraTree:
        """Create a Safra tree with only one node containing one state."""
        return cls({0: SafraNode(0, fset([initial_state]), ())})

    def power(self, buchi: BuchiAutomaton, input: int) -> SafraTree:
        """Apply the power set rule."""
        result = SafraTree()
        for name, node in self.items():
            result[name] = SafraNode(name,
                                     label=fset().union(*(buchi.next(s, input)
                                                          for s in node.label)),
                                     children=node.children,
                                     marked=False)

        return result

    def branch_accepting(self, accepting: set[Label]) -> SafraTree:
        """Apply the branch accepting rule."""

        available_names = [i for i in reversed(range(2 * len(self))) if i not in self]

        def get_nodes(name: int) -> Iterator[SafraNode]:
            node = self[name]

            acc = node.label.intersection(accepting)
            if acc:
                new_name = available_names.pop()
                yield dataclasses.replace(node, children=node.children + (new_name, ))
                yield SafraNode(new_name, acc, ())
            else:
                yield node
            for c in node.children:
                yield from get_nodes(c)

        return SafraTree({n.name: n for n in get_nodes(0)})

    def make_disjoint(self) -> SafraTree:
        """Apply the make disjoint rule."""

        def get_nodes(name: int, parent: SafraNode | None) -> Iterator[SafraNode]:
            node = self[name]

            if parent is None:
                new = node
            else:
                siblings = parent.children
                older_siblings = siblings[:siblings.index(name)]
                older_labels = fset().union(*(self[c].label for c in older_siblings))
                # We remove those that have been removed from the parent also
                new_label = node.label.difference(older_labels).intersection(parent.label)
                new = dataclasses.replace(node, label=new_label)

            yield new
            for c in node.children:
                yield from get_nodes(c, new)

        return SafraTree({n.name: n for n in get_nodes(0, None)})

    def mark_nodes(self) -> SafraTree:
        """Apply the mark nodes rule."""

        def get_nodes(name: int) -> Iterator[SafraNode]:
            node = self[name]

            if set().union(*(self[c].label
                             for c in node.children)) == node.label and node.label != fset():
                yield SafraNode(node.name, node.label, (), True)
            else:
                yield node
                for c in node.children:
                    yield from get_nodes(c)

        return SafraTree({n.name: n for n in get_nodes(0)})

    def remove_empty(self) -> SafraTree:
        result = SafraTree()
        for name, node in self.items():
            if node.label or node.name == 0:                                               # We allways keep the root
                result[name] = SafraNode(name, node.label,
                                         tuple(c for c in node.children if self[c].label),
                                         node.marked)
        return result

    def next(self, buchi: BuchiAutomaton, input: int) -> SafraTree:
        """Apply the 5 rules of the safra construction."""
        return (self.branch_accepting(buchi.accepting_states) # 2: create new children
                .power(buchi, input)                          # 1: First step
                .make_disjoint()                              # 3: Clean up
                .remove_empty()                               #    and remove empty nodes
                .mark_nodes())                                # 4: Mark children

    def next_latex_details(self, buchi: BuchiAutomaton, input: int) -> str:
        """Show the 5 steps of the construction in LaTeX."""

        def arrow(legend: str) -> str:
            l1, _, l2 = legend.partition(' ')
            if l2:
                legend = r"\substack{\text{%s}\\\text{%s}}" % (l1, l2)
            else:
                legend = r"\text{%s}" % l1
            return "$\\xto{" + legend + "}$"

        steps = [self.to_latex_forest()]
        s = self
        s = s.branch_accepting(buchi.accepting_states)
        steps += [arrow('Branch accepting'), s.to_latex_forest()]
        s = s.power(buchi, input)
        steps += [arrow('Power set'), s.to_latex_forest()]
        s = s.make_disjoint()
        steps += [arrow('Make disjoint'), s.to_latex_forest()]
        s = s.remove_empty()
        steps += [arrow('Remove empty'), s.to_latex_forest()]
        s = s.mark_nodes()
        steps += [arrow('Mark nodes'), s.to_latex_forest()]
        return '\n'.join(steps)

    def __str__(self) -> str:
        """Show the tree in a human readable way."""

        def node_str(name: int) -> str:
            node = self[name]
            label = '{' + ', '.join(map(str, node.label)) + '}'
            main = f"{name}: {label}"
            if node.marked:
                main += ' !'

            children = '\n'.join(map(node_str, node.children))
            if children:
                main += '\n' + indent(children, '    ')
            return main

        return node_str(0)

    def to_latex_forest(self) -> str:
        """Get the tree in LaTeX format."""

        def node_str(name: int) -> str:
            node = self[name]
            if node.label:
                label = ''.join(sorted(map(str, node.label)))
            else:
                label = '$\\emptyset$'
            full = "\\nodepart{one}" + str(name) + "\\nodepart{two}" + label
            if node.marked:
                full += ' !'
            full = "{" + full + "}"

            children = '\n'.join(map(node_str, node.children))
            if children:
                return f"[{full}\n{indent(children, '  ')}]"
            else:
                return f"[{full}]"

        r = "\\begin{forest}safra,\n"
        r += node_str(0)
        r += '\n\\end{forest}'
        return r


@dataclass
class Automaton:
    """Base class for automaton"""

    initial_state: Label

    def transitions(self) -> Generator[tuple[Label, int, Label], None, None]:
        raise NotImplementedError

    def to_graphviz(self, labels: dict[Label, str] | None = None, edge_len: float = 1.0) -> str:
        """
        Convert the automaton to the graphviz format.

        Arguments:
            labels: An optional dictionary of state labels to show.
            edge_len: The length of the edges for use with the neato algorithm.
        """

        r = f"digraph {self.__class__.__name__} {{\n" #rankdir=LR;\n"

        # Nodes / states
        if labels:
            r += "  {"
            for state, txt in labels.items():
                escaped = txt.replace('"', r'\"')
                initial = ',initial' if state == self.initial_state else ''
                r += f'    {state} [label="{escaped}",style="safrastate{initial}"];\n'
            r += "  }\n"

        # Edges / transitions
        r += f'edge [len={edge_len}];\n'
        for l in (0, 1):
            if l == 0:
                r += 'edge [style="edge0"];\n'
            else:
                r += 'edge [style="edge1"];\n'

            for start, label, end in self.transitions():
                if label == l:
                    r += f"{start} -> {end} [label=\"{label}\"];\n"

        r += "}\n"

        return r


@dataclass
class BuchiAutomaton(Automaton):
    """A non-deterministic Büchi automaton."""

    transition_function: Transition
    accepting_states: set[Label]

    def transitions(self) -> Generator[tuple[Label, int, Label], None, None]:
        for (s, i), next_states in self.transition_function.items():
            for next_state in next_states:
                yield s, i, next_state

    def next(self, state: Label, input: int) -> set[Label]:
        return self.transition_function.get((state, input), set())

    def to_muller(self) -> tuple[MullerAutomaton, dict[int, SafraTree]]:
        """Convert to a deterministic Muller automaton using the Safra construction"""

        start = SafraTree.new(self.initial_state)
        to_compute = deque([(start, 0), (start, 1)])
        transition: dict[tuple[Label, int], SafraTree] = {}
        ids: list[SafraTree] = []
        while to_compute:
            # print("left:", len(to_compute), "computed:", len(transition))
            tree, input = to_compute.popleft()
            try:
                id = ids.index(tree)
            except ValueError:
                id = len(ids)
                ids.append(tree)

            if (id, input) in transition:
                continue # already computed
            new = tree.next(self, input)
            transition[id, input] = new
            to_compute.append((new, 0))
            to_compute.append((new, 1))

        transition_with_ids: DetTransition = {
            key: ids.index(value)
            for key, value in transition.items()
        }

        return MullerAutomaton(0, transition_with_ids, []), dict(enumerate(ids))


@dataclass
class MullerAutomaton(Automaton):
    """A deterministic Muller automaton."""

    transition_function: DetTransition
    accepting_states: list[set[Label]]

    def transitions(self) -> Generator[tuple[Label, int, Label], None, None]:
        for (s, i), next_state in self.transition_function.items():
            yield s, i, next_state


def random_buchi(node_count: int, retries: int = 1) -> BuchiAutomaton:
    assert retries > 0

    def random_transition() -> set[Label]:
        nodes = list(range(node_count))
        shuffle(nodes)
        return set(nodes[:randrange(3)])

    size = 0
    # for _ in range(retries):
    while size != 180:
        buchi = BuchiAutomaton(0, {(i, k): random_transition()
                                   for i in range(node_count)
                                   for k in range(2)}, random_transition())
        _, labels = buchi.to_muller()
        if len(labels) > 100:
            best = buchi
            size = len(labels)
            pprint(buchi)
            print(size)
            yield buchi

        if len(labels) == 180:
            return buchi

    return best


A, B, C, D, E, F, G, Z = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z']
AUTOMATA: list[BuchiAutomaton] = [
    BuchiAutomaton(  # Infinitely many 1 but finitely many 11
        initial_state=A,
        transition_function={
            (A, 0): {A, B},
            (A, 1): {A, B},
            (B, 0): {B},
            (B, 1): {C},
            (C, 0): {B},
            (C, 1): set(),
        },
        accepting_states={C},
    ),
    BuchiAutomaton(  # Infinitely many 1 and iff infinitely many 11 then infinitely many 111
        transition_function={
            (A, 0): {A, B, D},
            (A, 1): {A, B, D},
            (B, 0): {B},
            (B, 1): {C},
            (C, 0): {B},
            (C, 1): set(),
            (D, 0): {D},
            (D, 1): {E},
            (E, 0): {D},
            (E, 1): {F},
            (F, 0): {D},
            (F, 1): {G},
            (G, 0): {D},
            (G, 1): {D, G},
        },
        accepting_states={C, G},
        initial_state=A,
    ),
    BuchiAutomaton(  # zeros always followed by the same parity of ones
        transition_function={
            (A, 0): {B},
            (A, 1): {C},
            (B, 0): {A},
            (B, 1): {D},
            (C, 0): set(),
            (C, 1): {D, Z},
            (D, 0): set(),
            (D, 1): {C},
            (Z, 0): {B},
            (Z, 1): set(),
        },
        accepting_states={Z},
        initial_state=A
    ),
    BuchiAutomaton(  # 4 states but produces 270 transitions in the Muller automtaton
        initial_state=0,
        transition_function={
            (0, 0): {0},
            (0, 1): {3},
            (1, 0): {0, 3},
            (1, 1): {1},
            (2, 0): {1},
            (2, 1): {0, 2},
            (3, 0): {2},
            (3, 1): set()},
        accepting_states={0}
    ),
BuchiAutomaton(initial_state=0,
               transition_function={(0, 0): {1},
                                    (0, 1): set(),
                                    (1, 0): {0},
                                    (1, 1): {2},
                                    (2, 0): {1, 2},
                                    (2, 1): {3},
                                    (3, 0): {3},
                                    (3, 1): {1}},
               accepting_states={0}),
    BuchiAutomaton(initial_state=0,
               transition_function={(0, 0): set(),
                                    (0, 1): {1},
                                    (1, 0): {0},
                                    (1, 1): {3},
                                    (2, 0): {0, 3},
                                    (2, 1): {2, 3},
                                    (3, 0): {0, 2},
                                    (3, 1): {0}},
               accepting_states={1}),
               BuchiAutomaton(initial_state=0,
               transition_function={(0, 0): set(),
                                    (0, 1): {3},
                                    (1, 0): {3},
                                    (1, 1): set(),
                                    (2, 0): {2},
                                    (2, 1): {0},
                                    (3, 0): {0},
                                    (3, 1): {1, 2}},
               accepting_states={1})
]


@click.command()
@click.option('-n', '--no-trees', is_flag=True, default=False, help="Don't draw the trees inside the nodes of the Muller automaton.")
@click.option('-s', '--safra-transition', type=str, default=None, help="Draw the steps to build a specific transition in the Muller automata. Pass a word of 0 and 1 and it will show the last transition when reading this word.")
@click.option('--prog', default='dot', type=str, help='The program to use to render the graph. "dot" and "neato" work best.')
@click.option('--edge-len', default=1.0, type=float, help="Target length for edges for the neato layout. [default=1.0]")
@click.option('-o', '--output', default='graph.tex', help="Latex file to store the graph to. [default=graph.tex]")
@click.option('-b', '--draw-buchi', is_flag=True, help="Draw the Buchi automaton instead of the Muller automton.")
@click.argument('automaton', default=-1)
def main(edge_len: float,
         no_trees: bool,
         safra_transition: str | None,
         prog: str,
         output: str,
         automaton: int,
         draw_buchi: bool = False) -> None:
    """Draw and convert a Buchi automaton into a Muller automaton.

    To specifiy which automaton to use, modify the AUTOMATA list in the code,
    and then set the AUTOMATON argument to the index of the automaton in the list.
    """

    buchi = AUTOMATA[automaton]

    if safra_transition is None:

        # for buchi in random_buchi(4):
            if draw_buchi:
                dot = buchi.to_graphviz()
            else:
                muller, safra_trees = buchi.to_muller()
                if no_trees:
                    labels = None
                else:
                    labels = {k: v.to_latex_forest() for k, v in safra_trees.items()}
                    print("Labels:", len(labels))
                dot = muller.to_graphviz(labels, edge_len)

            template = open('template.tex', 'r').read() + '\n'
            tex = dot2tex.dot2tex(
                dot,
                prog=prog,
                format='tikz',
                texmode='raw',
                crop=True,
                autosize=True,
                tikzedgelabels=True,
                template=template,
            )
            with open(output, 'w') as f:
                f.write(tex)
            subprocess.check_call(['pdflatex', '--output-directory', 'out/', output], stdout=subprocess.DEVNULL)

    else:
        safra = SafraTree({
            0: SafraNode(0, fset(buchi.initial_state), ()),
        })

        for i, letter in enumerate(safra_transition[:-1]):
            safra = safra.next(buchi, int(letter))

        print(r"""
\input{../preambule.tex}
\forestset{
    safra/.style={
        for tree={
            draw,
            rectangle split,
            rectangle split parts=2,
            rectangle split horizontal,
            % rectangle split part fill={red, white},
            fill=white,
            rounded corners,
            draw=black,
            anchor=north,
        },
    },
}

\tikzset{
    safrastate/.style={
        thin,circle,fill=ghostwhite,
    },
    edge0/.style={
        atomictangerine,
        ultra thick,
        every node/.style={
            draw, circle,
            text=black,
            fill=white,
        }
    },
    edge1/.style={
        bondiblue,
        ultra thick,
        every node/.style={
            draw, circle,
            text=black,
            fill=white,
        },
    }
}

\forestset{ default preamble={ for tree={draw,shape=rectangle split, rectangle split parts=2, rectangle split horizontal,anchor=north} } }
\begin{document}
        """)
        print(safra.next_latex_details(buchi, int(safra_transition[-1])))
        print(r"\end{document}")


if __name__ == "__main__":
    main()
