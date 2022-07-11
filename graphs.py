from dataclasses import dataclass
import matplotlib.pyplot as plt

# TODO
# - especially for tables format the numbers like {:.3} or something
# - join the tables somehow
import numpy as np


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
    legend cell align={left},
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
              "violet",
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
def features():
    print(convert_csv("Different features on scene 1",
                      """Accuracy	SIFT rectified 	SIFT unrectified	RootSIFT rectified	RootSIFT unrectified	HardNet rectified	HardNet unrectified	unrectified BRISK	unrectified SuperPoint	rectified BRISK	rectified SuperPoint
0	0.905	0.915	0.965	0.975	0.923	0.929	0.89	0.94	0.91	0.975
1	0.82	0.79	0.855	0.895	0.903	0.882	0.66	0.9	0.715	0.89
2	0.69	0.68	0.68	0.755	0.892	0.882	0.405	0.805	0.585	0.815
3	0.635	0.43	0.6	0.475	0.817	0.713	0.25	0.57	0.47	0.65
4	0.555	0.27	0.535	0.31	0.766	0.587	0.16	0.455	0.42	0.475
5	0.505	0.175	0.49	0.195	0.697	0.4	0.085	0.265	0.335	0.34
6	0.4	0.12	0.35	0.15	0.585	0.343	0.015	0.245	0.28	0.3
7	0.405	0.02	0.37	0.04	0.608	0.217	0.035	0.08	0.23	0.21
8	0.335	0.025	0.31	0.01	0.552	0.213	0	0.04	0.205	0.125
9	0.3	0.015	0.285	0.01	0.522	0.075	0.01	0.015	0.145	0.135
10	0.2	0	0.135	0	0.484	0.025	0.01	0	0.095	0.075
11	0.255	0	0.2	0	0.402	0	0.005	0	0.125	0.04
12	0.22	0	0.135	0	0.387	0	0.005	0	0.09	0.03
13	0.175	0	0.135	0.005	0.35	0.015	0.01	0	0.085	0.005
14	0.01	0.01	0.015	0	0.085	0	0	0	0	0
15	0	0.005	0.005	0	0.015	0.005	0	0	0.005	0
16	0	0	0	0	0	0.01	0	0	0	0
17	0.005	0.016	0	0	0.005	0.016	0	0	0	0"""))

#     print(convert_csv("Different feature descriptors",
#                       """Accuracy	unrectified BRISK	unrectified SuperPoint	rectified BRISK	rectified SuperPoint
# 0	0.89	0.94	0.91	0.975
# 1	0.66	0.9	0.715	0.89
# 2	0.405	0.805	0.585	0.815
# 3	0.25	0.57	0.47	0.65
# 4	0.16	0.455	0.42	0.475
# 5	0.085	0.265	0.335	0.34
# 6	0.015	0.245	0.28	0.3
# 7	0.035	0.08	0.23	0.21
# 8	0	0.04	0.205	0.125
# 9	0.01	0.015	0.145	0.135
# 10	0.01	0	0.095	0.075
# 11	0.005	0	0.125	0.04
# 12	0.005	0	0.09	0.03
# 13	0.01	0	0.085	0.005
# 14	0	0	0	0
# 15	0	0	0.005	0
# 16	0	0	0	0
# 17	0	0	0	0
#                       """))


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

# ablation = low
def ablation_mean_shift():
    print(convert_csv("refinement variants",  """Accuracy(5º)	mean_shitt_type=mean	mean_shift_type_None_feature_descriptor_SIFT_rectify_True	mean_shift_type_full_feature_descriptor_SIFT_rectify_True
0	0.905	0.925	0.89
1	0.82	0.785	0.815
2	0.69	0.69	0.7
3	0.635	0.57	0.63
4	0.555	0.515	0.57
5	0.505	0.425	0.48
6	0.4	0.355	0.38
7	0.405	0.36	0.39
8	0.335	0.245	0.345
9	0.3	0.24	0.28
10	0.2	0.16	0.22
11	0.255	0.145	0.245
12	0.22	0.065	0.225
13	0.175	0.015	0.185
14	0.01	0.005	0.01
15	0	0	0
16	0	0	0.005
17	0.005	0.005	0.005""", together=False, matching=True)[0])

# all_ds
def all_ds():
    print(convert_csv("test",  """	unrectified SIFT	rectified SIFT	AffNet shape estimators and depth maps
0	0.8928571429	0.8714285714	0.9528571429
1	0.6842857143	0.5971428571	0.9911428571
2	0.5192857143	0.4535714286	0.9138571429
3	0.3414285714	0.3314285714	0.8494285714
4	0.24	0.2542857143	0.7428571429
5	0.1678571429	0.2057142857	0.6637142857
6	0.09071428571	0.1507142857	0.5727142857
7	0.05214285714	0.12	0.5227142857
8	0.03214285714	0.1121428571	0.4742857143
9	0.01428571429	0.09	0.427
10	0.007857142857	0.05714285714	0.3952857143
11	0.002857142857	0.03	0.368
12	0.005	0.02071428571	0.3435714286
13	0.005775577558	0.008254125413	0.2145214521
14	0.004052156469	0.002006018054	0.1628786359
15	0	0	0.1191827469
16	0.002680965147	0.001340482574	0.06258579088
17	0	0	0.02031818182""", together=False, matching=True)[0])


# all_ds
def EVD():
    print(convert_csv("test",  """acc	SIFT rect	SIFT unrect	Hard Net rect	AffNet + depth	AffNet covering
1	0	0	0	0	0
2	0	0	0	1	2
3	0	0	0	2	5
5	0	0	1	4	8
10	0	0	2	7	9
20	0	0	2	9	10""", together=False, matching=True)[1])


def footest():

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


def plot_bar(interesting_keys, data, ylabel, title):

    value_index = 7

    interesting_keys_keys = [key[0] for key in interesting_keys]

    config_names = [d[0] for d in data]

    stats = [[0.0] * len(data) for _ in range(len(interesting_keys))]
    for conf_index, config_data in enumerate(data):
        #stats_line = [0.0 for i in range(len(interesting_keys_keys))]
        lines = config_data[1].split("\n")
        for line in lines:
            tokens = line.strip().split(" ")
            key = tokens[0]
            if key in interesting_keys_keys:
                data_index = interesting_keys_keys.index(key)
                value = float(tokens[value_index])
                if key == "matching":
                    value = value / 2.0
                stats[data_index][conf_index] = value

    stats_cum = [stats[0]]
    for i in range(1, len(stats)):
        stats_cum.append([stats_cum[i-1][j] + stats[i][j] for j in range(len(data))])

    print("Keys: {}".format(interesting_keys_keys))
    print("Stats: {}".format(stats))

    fig, ax = plt.subplots()
    width = 0.35

    for i in range(len(interesting_keys_keys)):
        data = stats[i]
        label = interesting_keys[i][1]
        if i == 0:
            ax.bar(config_names, data, width, label=label)
        else:
            print("label: {}; data: {}, bottom={}".format(label, data, stats_cum[i-1]))
            ax.bar(config_names, data, width, bottom=stats_cum[i-1], label=label)

    ax.set_ylim([0, 30])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.savefig("work/graph.pdf")
    plt.show()


def plot_bar_simple(method_times_data, ylabel, title=None):

    keys_s = set()
    keys = []
    for _, method in enumerate(method_times_data):
        for k in method[1]:
            if not keys_s.__contains__(k):
                keys_s.add(k)
                keys.append(k)

    # HACK
    cummulative_times = [[], []]
    times = [[], []]
    last_ct = [0.0, 0.0]
    for key in keys:
        for i, data in enumerate(method_times_data):
            if data[1].__contains__(key):
                last_ct[i] = last_ct[i] + data[1][key]
                times[i].append(data[1][key])
            else:
                times[i].append(0.0)
            cummulative_times[i].append(last_ct[i])

    print("cummulative_times: {}".format(cummulative_times))
    print("keys: {}".format(keys))

    fig, ax = plt.subplots(figsize=(5, 5))
    width = 0.10

    xpos = [0.2, 0.4]
    method_names = [mtd[0] for _, mtd in enumerate(method_times_data)]
    for i, key in enumerate(keys):
        data = [time[i] for time in times]
        if i == 0:
            ax.bar(xpos, data, align='center', width=width, label=key)
        else:
            cum_data = [time[i-1] for time in cummulative_times]
            print("label: {}; data: {}, bottom={}".format(key, method_times_data, cum_data))
            ax.bar(xpos, data, align='center', width=width, bottom=cum_data, label=key)

    axes_fontsize = 'xx-large'
    plt.xticks(xpos, method_names, fontsize=axes_fontsize)

    # ax.get_xaxis().set_visible(False)
    ax.set_ylim([0, 15])

    ax.set_ylabel(ylabel, fontsize=axes_fontsize)
    ax.set_title(title)
    legend_fontsize = 'x-large'
    ax.legend(fontsize=legend_fontsize)

    plt.savefig("work/running_time_comparison.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


@dataclass
class Stat:
    c: int
    avg: float
    avg_gr: float


def running_time_stacked_bars():

    data = [
        ["DepthAffNet",
         {"Rectification": 8.8,
          "Clustering": 2.5,
          "Depth maps": 0.65}],
        ["DenseAffNet",
             {"Rectification": 5.9558,
             "Clustering": 1.4904}],
            ]

    plot_bar_simple(data, 'Running time [s.]')

    #     data = [["dense AffNet",
    #     """
    # :affnet_clustering called 1421 times and it took 3.6307 secs. on average
    # :affnet_rectify called 1421 times and it took 6.6345 secs. on average
    # processing img from scratch called 1421 times and it took 10.2654 secs. on average
    # matching called 3590 times and it took 4.7471 secs. on average
    #     """],
    #
    #     ["MonoDepth + AffNet",
    #     """
    #     MonoDepth called 1421 times and it took 0.7
    # :affnet_rectify called 1421 times and it took 5.5726 secs. on average
    # :compute_normals_from_svd called 1421 times and it took 0.9551 secs. on average
    # sky masking called 1421 times and it took 0.5229 secs. on average
    # :cluster_normals called 1421 times and it took 0.0468 secs. on average
    # :possibly_upsample_early called 1421 times and it took 0.0047 secs. on average
    # :get_connected_components called 1421 times and it took 0.0821 secs. on average
    # processing img from scratch called 1421 times and it took 7.1972 secs. on average
    # matching called 3590 times and it took 5.9668 secs. on average
    #
    #     """]]

    # interesting_keys = [
    #     ["MonoDepth", "depth maps (MonoDepth)"],
    #     ["sky_masking", "sky semantic segmentation"],
    #     [":compute_normals_from_svd", "compute normals"],
    #     #[":get_connected_components", ""],
    #     [":affnet_rectify", "affine rectification"],
    #     [":affnet_clustering", "dense affnet"],  # ?
    #     ["matching", "matching/2"]]
    #
    # plot_bar(data, 'Computation complexity', 'dense AffNet vs. MonoDepth + AffNet computation complexity', interesting_keys=interesting_keys)


def bar_plot_example():

    config_names = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 35, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    ch_means = [20, 35, 30, 35, 27]

    mw = [men_means[i] + women_means[i] for i in range(len(men_means))]

    men_std = [2, 3, 4, 1, 2]
    women_std = [3, 5, 2, 3, 3]
    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(config_names, men_means, width, label='Men')
    ax.bar(config_names, women_means, width, bottom=men_means, label='Women')
    ax.bar(config_names, ch_means, width, bottom=mw, label='Children')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.legend()

    plt.show()


def graph_grid():

    x = [1, 2, 3, 4]
    y = [234, 124, 368, 343]

    fig = plt.figure(1, figsize=(8, 6))
    fig.suptitle('Example Of Plot With Grid Lines')

    plt.plot(x, y)

    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)

    plt.minorticks_on()
    plt.grid(b=False, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    # EVD()
    running_time_stacked_bars()
    # bar_plot_example()
    # graph_grid()