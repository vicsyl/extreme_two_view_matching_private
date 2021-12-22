from dataclasses import dataclass


@dataclass
class Style:
    color: str
    mark: str
    entry_name: str


#%overleaf.com/learn/latex/Pgfplots_package

def convert_to_graph(title, style_data_list):

    beginning_template = """\\begin{tikzpicture}
    \\begin{axis}[
        width=\\textwidth,
        height=8cm,
        % scale only axis,
        title={""" + title + """},
        xlabel={Difficulty},
        ylabel={Accuracy (error < 5$^{\\circ}$)},
        xmin=0, xmax=17,
        ymin=0, ymax=1,
        xtick={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17},
        ytick={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1},
        legend pos=north east,
        ymajorgrids=true,
        xmajorgrids=true,
        grid style=dashed,
    ]
    """

    for plot in style_data_list:
        style: Style = plot[0]
        data = plot[1]
        data_str = " ".join(["({},{})".format(str(d[0]), str(d[1])) for d in data])
        beginning_template = beginning_template + """
        \\addplot[
        color=""" + style.color + """,
        mark=""" + style.mark + """,
        ]
        coordinates { """ + data_str + """
        };
        \\addlegendentry{""" + style.entry_name + "}\n"

    beginning_template = beginning_template + """
    \\end{axis}
\\end{tikzpicture}"""

    return beginning_template


def convert_csv(title, csv_in_str):

    colors = ["red",
              "green",
              "yellow,",
              "blue",
              "black"]

    leave_out_1st = False
    for line in csv_in_str.splitlines():
        tokens = line.split("\t")
        floats = [float(token) for token in tokens if len(token) > 0]
        if len(floats) == 0:
            continue
        leave_out_1st |= max(floats) > 1.0
        length = len(floats)

    length = length - 1 if leave_out_1st else length

    style_data_list = [None] * length
    for i in range(length):
        style_data_list[i] = (Style(color=colors[i % len(colors)], mark="square", entry_name="entry{}".format(i)), [])

    for diff, line in enumerate(csv_in_str.splitlines()):
        tokens = line.split("\t")
        tokens_to_read = tokens[-length:]
        for i, t in enumerate(tokens_to_read):
            style_data_list[i][1].append((float(diff), t))

    return convert_to_graph(title, style_data_list)


def main():

    print("Basic use case")
    print(convert_to_graph("hn new title", [
        (Style(color="red", mark="square", entry_name="hn1"), ((0, 0.9), (1, 0.7))),
        (Style(color="blue", mark="square", entry_name="hn2"), ((0, 0.9), (1, 0.6))),
    ]))

    print("Converting from csv")
    print(convert_csv("HN title", """
    0	0.935
    1	0.925
    2	0.905
    3	0.88
    4	0.865
    5	0.71
    6	0.575
    7	0.615
    8	0.575
    9	0.545
    10	0.535
    11	0.42
    12	0.4
    13	0.345
    14	0.125
    15	0.05
    16	0.02
    17	0.005"""))

    print("Converting from csv 2")
    print(convert_csv("HN title 2",
"""0	0.935	0.955	0.923	0.905
1	0.925	0.905	0.903	0.82
2	0.905	0.885	0.892	0.69
3	0.88	0.865	0.817	0.635
4	0.865	0.825	0.766	0.555
5	0.71	0.645	0.697	0.505
6	0.575	0.53	0.585	0.4
7	0.615	0.575	0.608	0.405
8	0.575	0.555	0.552	0.335
9	0.545	0.505	0.522	0.3
10	0.535	0.535	0.484	0.2
11	0.42	0.425	0.402	0.255
12	0.4	0.41	0.387	0.22
13	0.345	0.3	0.35	0.175
14	0.125	0.085	0.085	0.01
15	0.05	0.06	0.015	0
16	0.02	0.005	0	0
17	0.005	0	0.005	0.005"""))


if __name__ == '__main__':
    main()