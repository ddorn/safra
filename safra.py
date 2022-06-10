from __future__ import annotations
from collections import deque

from dataclasses import dataclass
import dataclasses
import subprocess
import dot2tex
import click
from pprint import pprint
from textwrap import indent
from typing import Generator, Generic, Iterator, TypeVar


Label = TypeVar('Label')
fset = frozenset


@dataclass(frozen=True)
class SafraNode(Generic[Label]):
    """
    A node in the Safra tree.
    """
    name: int
    label: fset[Label]
    children: tuple[int, ...]
    marked: bool = False


class SafraTree(dict[int, SafraNode[Label]]):
    """
    A Safra tree, where each node is labeled with an integer.
    """

    @classmethod
    def new(cls, initial_state: Label) -> SafraTree[Label]:
        return cls({0: SafraNode(0, fset([initial_state]), ())})

    def power(self, buchi: BuchiAutomaton[Label], input: int) -> SafraTree[Label]:
        result = SafraTree[Label]()
        for name, node in self.items():
            result[name] = SafraNode(name,
                                     label=fset().union(*(buchi.next(s, input)
                                                          for s in node.label)),
                                     children=node.children,
                                     marked=False)

        return result

    def branch_accepting(self, accepting: set[Label]) -> SafraTree[Label]:

        available_names = [i for i in reversed(range(2 * len(self))) if i not in self]

        def get_nodes(name: int) -> Iterator[SafraNode[Label]]:
            node = self[name]

            acc = node.label.intersection(accepting)
            if acc:
                new_name = available_names.pop()
                yield dataclasses.replace(node, children=node.children + (new_name, ))
                yield SafraNode[Label](new_name, acc, ())
            else:
                yield node
            for c in node.children:
                yield from get_nodes(c)

        return SafraTree[Label]({n.name: n for n in get_nodes(0)})

    def make_disjoint(self) -> SafraTree[Label]:

        def get_nodes(name: int, parent: int | None) -> Iterator[SafraNode[Label]]:
            node = self[name]

            if parent is None:
                yield node
            else:
                siblings = self[parent].children
                older_siblings = siblings[:siblings.index(name)]
                yield dataclasses.replace(node,
                                          label=node.label.difference(*(self[s].label
                                                                        for s in older_siblings)))

            for c in node.children:
                yield from get_nodes(c, name)

        return SafraTree({n.name: n for n in get_nodes(0, None)})

    def union_children(self) -> SafraTree[Label]:

        def get_nodes(name: int) -> Iterator[SafraNode[Label]]:
            node = self[name]

            if set().union(*(self[c].label
                             for c in node.children)) == node.label and node.label != fset():
                yield SafraNode[Label](node.name, node.label, (), True)
            else:
                yield node
                for c in node.children:
                    yield from get_nodes(c)

        return SafraTree[Label]({n.name: n for n in get_nodes(0)})

    def remove_empty(self) -> SafraTree[Label]:
        result = SafraTree[Label]()
        for name, node in self.items():
            if node.label or node.name == 0:                                               # We allways keep the root
                result[name] = SafraNode(name, node.label,
                                         tuple(c for c in node.children if self[c].label),
                                         node.marked)
        return result

    def next(self, buchi: BuchiAutomaton[Label], input: int) -> SafraTree[Label]:
        return (self
                .branch_accepting(buchi.accepting_states) # 2: create new children
                .power(buchi, input)                  # 1: First step
                .make_disjoint()                          # 3: Clean up
                .remove_empty()                           #    and remove empty nodes
                .union_children())                        # 4: Mark children

    def next_latex_details(self, buchi: BuchiAutomaton[Label], input: int) -> str:

        def arrow(legend: str) -> str:
            l1, _, l2 = legend.partition(' ')
            if l2:
                legend = r"\substack{\text{%s}\\\text{%s}}" % (l1, l2)
            else:
                legend = r"\text{%s}" % l1
            return "$\\xto{" + legend + "}$"

        steps = [self.to_latex_forest()]
        a = self
        a = a.branch_accepting(buchi.accepting_states)
        steps += [arrow('Branch accepting'), a.to_latex_forest()]
        a = a.power(buchi, input)
        steps += [arrow('Power set'), a.to_latex_forest()]
        a = a.make_disjoint()
        steps += [arrow('Make disjoint'), a.to_latex_forest()]
        a = a.remove_empty()
        steps += [arrow('Remove empty'), a.to_latex_forest()]
        a = a.union_children()
        steps += [arrow('Mark nodes'), a.to_latex_forest()]
        return '\n'.join(steps)

    def __str__(self) -> str:

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
class Automaton(Generic[Label]):
    """Base class for automaton"""

    initial_state: Label

    def transition(self) -> Generator[tuple[Label, int, Label], None, None]:
        raise NotImplementedError

    def to_graphviz(self, labels: dict[Label, str] | None = None, edge_len: float = 1.0) -> str:

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

            for start, label, end in self.transition():
                if label == l:
                    r += f"{start} -> {end} [label=\"{label}\"];\n"

        r += "}\n"

        return r


@dataclass
class BuchiAutomaton(Automaton[Label]):
    """A non-deterministic Büchi automaton."""

    transition_function: dict[tuple[Label, int], set[Label]]
    accepting_states: set[Label]

    def transition(self) -> Generator[tuple[Label, int, Label], None, None]:
        for (s, i), next_states in self.transition_function.items():
            for next_state in next_states:
                yield s, i, next_state

    def next(self, state: Label, input: int) -> set[Label]:
        return self.transition_function.get((state, input), set())

    def to_muller(self) -> tuple[MullerAutomaton[int], dict[int, SafraTree[Label]]]:
        """Convert to a deterministic Muller automaton using the Safra construction"""

        start = SafraTree.new(self.initial_state)
        to_compute = deque([(start, 0), (start, 1)])
        transition = {}
        ids: list[SafraTree[Label]] = []
        while to_compute:
            print("left:", len(to_compute), "computed:", len(transition))
            tree, input = to_compute.popleft()
            try:
                id = ids.index(tree)
            except ValueError:
                id = len(ids)
                ids.append(tree)

            if (id, input) in transition:
                continue
            new = tree.next(self, input)
            transition[id, input] = new
            to_compute.append((new, 0))
            to_compute.append((new, 1))

        transition_with_ids = {key: ids.index(value) for key, value in transition.items()}

        return MullerAutomaton(0, transition_with_ids, []), dict(enumerate(ids))


@dataclass
class MullerAutomaton(Automaton[Label]):
    """A deterministic Muller automaton."""

    transition_function: dict[tuple[Label, int], Label]
    accepting_states: list[set[Label]]

    def transition(self) -> Generator[tuple[Label, int, Label], None, None]:
        for (s, i), next_state in self.transition_function.items():
            yield s, i, next_state


A, B, C, D, E, F, G, Z = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z']
AUTOMATA: list[BuchiAutomaton[str]] = [
    BuchiAutomaton(
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
    BuchiAutomaton(
        transition_function={
            (A, 0): {A, B, D},
            (A, 1): {A, B, D},
            (B, 0): {B},
            (B, 1): {C},
            (C, 0): {B},
            (C, 1): {D, G},
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
    BuchiAutomaton(
        transition_function={
            (A, 0): {A, B},
            (A, 1): {A},
            (B, 0): {B},
            (B, 1): set(),
        },
        accepting_states={B},
        initial_state=A,
    ),
    BuchiAutomaton(
        transition_function={
            (A, 0): {B},
            (A, 1): set(),
            (B, 0): {C},
            (B, 1): set(),
            (C, 0): {B},
            (C, 1): {A},
        },
        accepting_states={C},
        initial_state=A,
    ),
    BuchiAutomaton(
        transition_function={
            (A, 0): {B},
            (A, 1): {D},
            (B, 0): {B},
            (B, 1): {C, D},
            (C, 0): set(),
            (C, 1): {C},
            (D, 0): {B},
            (D, 1): {D},
        },
        accepting_states={A, B, C},
        initial_state=A,
    ),
    BuchiAutomaton(
        transition_function={
            (A, 0): {A},
            (A, 1): {A, B},
            (B, 0): {B},
            (B, 1): set(),
        },
        accepting_states={B},
        initial_state=A,
    ),
    BuchiAutomaton(
        transition_function={
            (Z, 0): {A},
            (Z, 1): {},
            (A, 0): {Z, A, B},
            (A, 1): {B},
            (B, 0): {C},
            (B, 1): {A},
            (C, 0): {Z},
            (C, 1): {},
        },
        accepting_states={Z, C},
        initial_state=Z,
    ),
    BuchiAutomaton(  # zeros always followed by the same parity of ones
        transition_function={
            (A, 0): {B},
            (A, 1): {C},
            (B, 0): {A},
            (B, 1): {D},
            (C, 0): {},
            (C, 1): {D, Z},
            (D, 0): {},
            (D, 1): {C},
            (Z, 0): {B},
            (Z, 1): {},
        },
        accepting_states={Z},
        initial_state=A
    ),
    BuchiAutomaton(
        transition_function={
            (A, 0): {},
            (A, 1): {C},
            (B, 0): {},
            (B, 1): {A, B},
            (C, 0): {A, B},
            (C, 1): {},
        },
        accepting_states={B},
        initial_state=C,
    )
]


@click.command()
@click.option('--edge-len', default=1.0, type=float)
@click.option('-n', '--no-trees', is_flag=True, default=False)
@click.option('-s', '--safra-transition', type=str, default=None)
@click.option('--prog', default='dot', type=str)
@click.option('-o', '--output', default='graph.tex')
@click.option('-b', '--draw-buchi', is_flag=True)
@click.argument('automaton', default=-1)
def main(edge_len: float, no_trees: bool, safra_transition: str | None, prog: str, output: str,
         automaton: int, draw_buchi: bool = False) -> None:

    buchi = AUTOMATA[automaton]

    if safra_transition is None:
        muller, safra_trees = buchi.to_muller()
        if no_trees:
            labels = None
        else:
            labels = {k: v.to_latex_forest() for k, v in safra_trees.items()}

        # print(buchi.to_graphviz())
        if draw_buchi:
            dot = buchi.to_graphviz()
        else:
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
        subprocess.check_call(['pdflatex', '--output-directory', 'out/', output])

    else:
        safra = SafraTree[str]({
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
