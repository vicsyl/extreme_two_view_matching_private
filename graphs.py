from dataclasses import dataclass

# TODO
# - especially for tables format the numbers like {:.3} or something
# - join the tables somehow



@dataclass
class Style:
    color: str
    mark: str
    entry_name: str


#%overleaf.com/learn/latex/Pgfplots_package

def underscores_to_spaces(s):
    return s.replace("_", " ")


def convert_to_graph(title, style_data_list, together=True, matching=True):

    matching_ticks = """xlabel={Difficulty},
    ylabel={Accuracy (error in estimated relative rotation < 5$^{\\circ}$)},
    xmin=0, xmax=17,
    ymin=0, ymax=1,
    xtick={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17},
    ytick={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1},"""
    normals_ticks = """xlabel={parameter $\\alpha_{\\theta}$ [deg.]},
    ylabel={Absolute difference from the right angle [deg.]},
    xmin=15, xmax=35,
    ymin=0, ymax=15,
    xtick={15,20,25,30,35},
    ytick={0,2,4,6,8,10,12,14},"""
    ticks = matching_ticks if matching else normals_ticks

    def get_x_label(data):

        label_map = {0: 15, 1:20, 2:25, 3:30, 4:35}

        if matching:
            return str(data)
        else:
            return str(label_map[data])


    graph = """
\\begin{figure}[H]
\\centering
\\begin{tikzpicture}
\\begin{axis}[
    width=\\textwidth,
    height=8cm,
    % scale only axis,
    title={""" + underscores_to_spaces(title) + """},
    """ + ticks + """
    legend pos=north east,
    ymajorgrids=true,
    xmajorgrids=true,
    grid style=dashed,
]
"""

    for plot in style_data_list:
        style: Style = plot[0]
        data = plot[1]
        data_str = " ".join(["({},{})".format(get_x_label(d[0]), str(d[1])) for d in data])
        graph = graph + """
        \\addplot[
        color=""" + style.color + """,
        mark=""" + style.mark + """,
        ]
        coordinates { """ + data_str + """
        };
        \\addlegendentry{""" + underscores_to_spaces(style.entry_name) + "}\n"


    graph = graph + """
    \\end{axis}
\\end{tikzpicture}
\\caption{""" + title + """} 
\\label{fig:""" + title[:10].lower().replace(" ", "_") + """}
\\end{figure}
"""

    table = """ 
\\begin{table}[h!]
\\begin{center}
\\begin{tabular}{|| """ + "|".join([" c " for _ in range(len(style_data_list) + 1)]) + """||} 
 \\hline
difficulty & """ + " & ".join([str(sd[0].entry_name.replace("_", "\\_")) for sd in style_data_list]) + """\\\\
\\hline
"""
    diffs = len(style_data_list[0][1])
    for diff in range(diffs):
        table = table + str(diff) + " & "
        table = table + " & ".join([str(sd[1][diff][1]) for sd in style_data_list]) + """\\\\
\\hline
"""
    table = table + """
\\end{tabular}
\\end{center}
\\caption{""" + title + """} 
\\end{table}
"""

# \\label{table:tabular_""" + title.replace(" ", "\\_") + """}
    if together:
        return graph + "\n\n" + table
    else:
        return graph, table


def convert_from_data(title, entries_names, diff_acc_data_lists, together=True, matching=True):

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

    return convert_to_graph(title, style_data_list, together, matching)


def is_numeric_my(s):
    set1 = set(s.strip())
    return len(set1) > 0 and set1.issubset(set(list("0123456789.")))


def convert_csv(title, csv_in_str, together=True, matching=True):

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

    return convert_from_data(title, entries, diff_acc_data_lists, together, matching)


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

# ablation
def ablation_high_svd_weighting():
    print(convert_csv("SVD weighting on scene 1",
                      """Accuracy (5º)	plain_SVD	weighted_SVD
0	0.9	0.905
1	0.8	0.82
2	0.66	0.69
3	0.61	0.635
4	0.545	0.555
5	0.47	0.505
6	0.395	0.4
7	0.39	0.405
8	0.285	0.335
9	0.255	0.3
10	0.17	0.2
11	0.2	0.255
12	0.105	0.22
13	0.09	0.175
14	0	0.01
15	0	0
16	0	0
17	0	0.005""", together=False)[0])


# ablation
def ablation_high_handle_ap():
    print(convert_csv("Handling of antipodal points on scene 1",
                      """Accuracy (5º)	with_antipodal_points_handling	without_antipodal_points_handling
0	0.91	0.905
1	0.795	0.82
2	0.665	0.69
3	0.625	0.635
4	0.53	0.555
5	0.475	0.505
6	0.38	0.4
7	0.41	0.405
8	0.33	0.335
9	0.29	0.3
10	0.19	0.2
11	0.22	0.255
12	0.18	0.22
13	0.12	0.175
14	0.01	0.01
15	0	0
16	0	0
17	0.016	0.005""", together=False)[0])


# ablation = low
def ablation_low_quantile():
    print(convert_csv("Different quantiles for filtering based on singular values ratio",  """	quantile=1.0	quantile=0.8	quantile=0.6	quantile=0.4
15	8.708	8.87	8.358	9.975
20	6.08	5.869	5.241	5.635
25	4.835	5.303	5.249	5.828
30	5.032	5.572	6.308	7.087
35	3.78	4.084	4.565	5.52""", together=False, matching=False)[0])


# ablation = low
def ablation_low_mean_shift():
    print(convert_csv("Different refinements of the initial bucket centers",  """	mean	mean-shift	no refinement
15	8.708	9.079	8.486
20	6.08	7.338	5.999
25	4.835	5.594	6.276
30	5.032	4.54	8.714
35	3.78	3.82	1.408""", together=False, matching=False)[0])


# ablation = low
def ablation_low_sigma():
    print(convert_csv("Different values of $\\sigma$ for SVD weighting",  """	$\\sigma$=0.6	$\\sigma$=0.8	$\\sigma$=1.0	$\\sigma$=1.2
15	12.71	8.708	7.756	7.493
20	10.686	6.08	5.202	5.215
25	9.648	4.835	5.402	6.561
30	7.761	5.032	6.531	7.968
35	5.772	3.78	5.074	6.223""", together=False, matching=False)[0])


# ablation = low
def ablation_low_ap():
    print(convert_csv("Handling of antipodal points",  """	antipodal points off	antipodal points on
15	8.708	10.479
20	6.08	6.247
25	4.835	5.04
30	5.032	4.811
35	3.78	3.23""", together=False, matching=False)[0])


# ablation = low
def ablation_low_svd_weighting():
    print(convert_csv("simple SVD and weighted SVD",  """	weighted svd	unweighted svd
15	8.708	8.261
20	6.08	7.37
25	4.835	9.728
30	5.032	11.437
35	3.78	9.087""", together=False, matching=False)[0])


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
    ablation_low_svd_weighting()