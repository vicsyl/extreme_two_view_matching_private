from dataclasses import dataclass


@dataclass
class Style:
    color: str
    mark: str
    entry_name: str


#%overleaf.com/learn/latex/Pgfplots_package

def convert_to_graph(title, style_data_list):

    beginning_template = """
\\begin{figure}[H]
\\centering
\\begin{tikzpicture}
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
\\end{tikzpicture}
\\caption{""" + title + """} 
\\label{fig:""" + title[:10].lower().replace(" ", "_") + """}
\\end{figure}
"""

    return beginning_template


def convert_from_data(title, entries_names, diff_acc_data_lists):

    # NOTE: cycle list

    colors = ["red",
              "blue",
              "green",
              "black",
              "magenta"]

    #https://www.iro.umontreal.ca/~simardr/pgfplots.pdf
    marks = [
        "triangle", "diamond", "asterisk", "square", "o", "|"
    ]

    style_data_list = []
    for i, entry_name in enumerate(entries_names):
        style_data_list.append((Style(color=colors[i % len(colors)], mark=marks[i % len(marks)], entry_name=entry_name), []))

    for i, diff_acc_data_list in enumerate(diff_acc_data_lists):
        for diff_acc_data in diff_acc_data_list:
            style_data_list[i][1].append((float(diff_acc_data[0]), diff_acc_data[1]))

    return convert_to_graph(title, style_data_list)


def is_numeric_my(s):
    set1 = set(s.strip())
    return len(set1) > 0 and set1.issubset(set(list("0123456789.")))


def convert_csv(title, csv_in_str):

    leave_out_1st_column = False
    leave_out_1st_row = False
    entries = None

    for line in csv_in_str.splitlines():

        if len(line.strip()) == 0:
            continue

        tokens = line.split("\t")
        if len(tokens) > 0 and not is_numeric_my(tokens[0]):
            if leave_out_1st_row:
                print("WARNING: leave_out_1st_row already True")
            else:
                entries = tokens
                leave_out_1st_row = True
            continue

        # these two checks are to handle an empty line
        floats = [float(token) for token in tokens if len(token) > 0]
        if len(floats) == 0:
            continue
        leave_out_1st_column |= max(floats) > 1.0
        length = len(floats)

    start = 1 if leave_out_1st_column else 0
    if entries is None:
        entries = [str(i) for i in range(length - start)]
    else:
        entries = entries[start:]
    diff_acc_data_lists = [[] for _ in entries]

    rel_lines = list(csv_in_str.splitlines())
    if leave_out_1st_row:
        rel_lines = rel_lines[1:]
    for diff, line in enumerate(rel_lines):
        tokens = line.split("\t")
        # TODO to float to 2 (3,4?) to string
        tokens_to_read = [token.strip() for token in tokens[start:]]
        for i, t in enumerate(tokens_to_read):
            if len(t) > 0:
                diff_acc_data_lists[i].append((float(diff), t))

    return convert_from_data(title, entries, diff_acc_data_lists)


# K 554
def all_unrectified_from_csv():
    print(convert_csv("Different versions in the whole Toft dataset",
                      """0.894375
                        0.699375
                        0.535625
                        0.349375
                        0.240625
                        0.1675
                        0.088125
                        0.046875
                        0.029375
                        0.0125
                        0.006875
                        0.0025
                        0.004375
                        0.004957507082
                        0.004210526316
                        0
                        0.002114164905
                        0.007841269841"""))

# M35
def morphology_csv():
    print(convert_csv("Morphological variations",
                      """unrectified	8-p con., flood fill, closing	8-p con.	4-p con.
0	0.915	0.905	0.922	0.935
1	0.79	0.8	0.756	0.82
2	0.68	0.66	0.647	0.688
3	0.43	0.58	0.527	0.59
4	0.27	0.505	0.423	0.538
5	0.175	0.47	0.4	0.435
6	0.12	0.365	0.414	0.359
7	0.02	0.38	0.436	0.333
8	0.025	0.255	0.294	0.246
9	0.015	0.275	0.295	0.256
10	0	0.155	0.141	0.136
11	0	0.175	0.131	0.151
12	0	0.1	0.105	0.092
13	0	0.035	0.05	0.04
14	0.01	0	0	0.01
15	0.005	0	0.005	0
16	0	0.005	0.005	0.005
17	0.016	0.011	0.012	0.011
                      """))


# B373
def feature_descriptors():
    print(convert_csv("Different feature descriptors",
                      """Accuracy	unrectified BRISK	unrectified SuperPoint	rectified BRISK	rectified SuperPoint
0	0.89	0.94	0.91	0.975
1	0.66	0.9	0.715	0.89
2	0.405	0.805	0.585	0.815
3	0.25	0.57	0.47	0.65
4	0.16	0.455	0.42	0.475
5	0.085	0.265	0.335	0.34
6	0.015	0.245	0.28	0.3
7	0.035	0.08	0.23	0.21
8	0	0.04	0.205	0.125
9	0.01	0.015	0.145	0.135
10	0.01	0	0.095	0.075
11	0.005	0	0.125	0.04
12	0.005	0	0.09	0.03
13	0.01	0	0.085	0.005
14	0	0	0	0
15	0	0	0.005	0
16	0	0	0	0
17	0	0	0	0
                      """))


# B483
def affnet_variants():
    print(convert_csv("HardNet and Affnet variants",
                      """Scene 1	AfffNet driven	HardNet rectified via H - baseline	baseline SIFT rectified 	AffNet driven better covering 0.9/2	AffNet driven better covering (0.95/3)	last_version - not the best
0	0.935	0.923	0.905	0.955	0.96	0.93
1	0.925	0.903	0.82	0.91	0.935	0.895
2	0.905	0.892	0.69	0.91	0.905	0.885
3	0.88	0.817	0.635	0.905	0.86	0.87
4	0.865	0.766	0.555	0.844	0.849	0.815
5	0.71	0.697	0.505	0.714	0.724	0.63
6	0.575	0.585	0.4	0.6	0.63	0.58
7	0.615	0.608	0.405	0.615	0.64	0.525
8	0.575	0.552	0.335	0.57	0.585	0.45
9	0.545	0.522	0.3	0.555	0.545	0.45
10	0.535	0.484	0.2	0.545	0.57	0.435
11	0.42	0.402	0.255	0.425	0.455	0.295
12	0.4	0.387	0.22	0.405	0.425	0.185
13	0.345	0.35	0.175	0.36	0.335	0.11
14	0.125	0.085	0.01	0.145		0.01
15	0.05	0.015	0	0.065		0.04
16	0.02	0	0	0.01		0.005
17	0.005	0.005	0.005	0.032		0"""))


# B529
def last_affnet_variants():
    print(convert_csv("HardNet variants",
                      """Accuracy (5º)	SIFT baseline 	homography rectification	naive, th=0.95	dense, th=0.9	dense, th=0.95	sparse, th=0.95	SIIM dense, th=0.95
0	0.905	0.923	0.965	0.955	0.952	0.954	0.935
1	0.82	0.903	0.915	0.899	0.928	0.934	0.91
2	0.69	0.892	0.9	0.899	0.907	0.887	0.89
3	0.635	0.817	0.86	0.84	0.863	0.889	0.905
4	0.555	0.766	0.845	0.869	0.876	0.871	0.895
5	0.505	0.697	0.725	0.673	0.686	0.738	0.765
6	0.4	0.585	0.635	0.595	0.592	0.644	0.64
7	0.405	0.608	0.625	0.625	0.645	0.651	0.65
8	0.335	0.552	0.585	0.55	0.583	0.583	0.605
9	0.3	0.522	0.555	0.54	0.562	0.577	0.58
10	0.2	0.484	0.57	0.55	0.583	0.593	0.585
11	0.255	0.402	0.475	0.5	0.485	0.525	0.51
12	0.22	0.387	0.42		0.44	0.485	0.48
13	0.175	0.35	0.335		0.385	0.435	0.415
14	0.01	0.085	0.145		0.175	0.185	0.13
15	0	0.015	0.07		0.1	0.12	0.07
16	0	0	0.005		0.04	0.035	0.016
17	0.005	0.005	0.011		0.032	0.026"""))


# O555
def affnet_2_major_variants():
    print(convert_csv("HardNet variants",
                      """sparse, th=0.95	SIIM dense, th=0.95
0.954	0.935
0.934	0.91
0.887	0.89
0.889	0.905
0.871	0.895
0.738	0.765
0.644	0.64
0.651	0.65
0.583	0.605
0.577	0.58
0.593	0.585
0.525	0.51
0.485	0.48
0.435	0.415
0.185	0.13
0.12	0.07
0.035	0.016
0.026	"""))


def test():

    print("Basic use case")
    print(convert_to_graph("hn new title", [
        (Style(color="red", mark="square", entry_name="hn1"), ((0, 0.9), (1, 0.7))),
        (Style(color="blue", mark="square", entry_name="hn2"), ((0, 0.9), (1, 0.6))),
    ]))

    print("Converting from csv")
    print(convert_csv("HN title",
                      """0	0.935
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
    affnet_2_major_variants()