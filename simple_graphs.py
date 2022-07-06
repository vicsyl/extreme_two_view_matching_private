import matplotlib.pyplot as plt
from graphs import is_numeric_my


def convert_csv(title, csv_in_str, file_name,
                difficulties=range(18),
                y_ticks=[float(i)/10 for i in range(11)],
                legend_fontsize='x-large'):

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot:~:text=Notes-,Format,-Strings
    # markers: . , o v ^ > < s p * + x d | _
    # line styles: - -- -. :
    # fmt=[marker][color][line]
    styles = [
        "bo-", "ro-", "yo-", "co-", "ko-"
        #"bo-", "ro+--", "gx-.", "c*-", "ko-"
    ]

    # let's assume True for both
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
    acc_data_lists_x = [[] for _ in entries]
    acc_data_lists_y = [[] for _ in entries]

    rel_lines = list(csv_in_str.splitlines())
    if leave_out_1st_row:
        rel_lines = rel_lines[1:]
    for diff, line in enumerate(rel_lines):
        tokens = line.split("\t")
        # TODO to float to 2 (3,4?) to string
        # tokens_to_read = []
        for i, t in enumerate(tokens[start:]):
            if len(t) > 0:
                acc_data_lists_x[i].append(float(diff))
                acc_data_lists_y[i].append(float(t.strip()))

    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle(title)

    for i, _ in enumerate(acc_data_lists_x):
        ax.plot(acc_data_lists_x[i], acc_data_lists_y[i], styles[i % len(styles)], label=entries[i])

    l_d = list(difficulties)
    #ax.set_xlim([0, len(acc_data_lists_x[0]) - 1])
    ax.set_xlim([l_d[0], l_d[-1]])
    ax.set_xticks(difficulties)
    ax.set_ylim(min(y_ticks), max(y_ticks))
    ax.set_yticks(y_ticks)
    #ax.set_xticks(range(0, len(acc_data_lists_x[0])))
    #plt.axis('auto')
    ax.legend(shadow=True, framealpha=None, fontsize=legend_fontsize)

    ax.set_xlabel("Difficulty", fontsize=legend_fontsize)
    ax.set_ylabel("Accuracy (relative rotation error < 5°)", fontsize=legend_fontsize)

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=False, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(file_name)
    plt.show()


    print()
    print(title, entries, acc_data_lists_x)


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


def scene1():

    convert_csv(
        #"Dense Affnet vs. MonoDepth+AffNet on scene 1 of the Strong ViewPoint Changes Dataset",
        "Accuracy on scene 1",
"""	dense AffNet	MonoDepth+AffNet	homography recttification	AffNet	HardNet unrectified
0	0.95	0.945	0.923	0.935	0.929
1	0.925	0.94	0.903	0.91	0.882
2	0.89	0.91	0.892	0.89	0.882
3	0.87	0.92	0.817	0.905	0.713
4	0.85	0.88	0.766	0.895	0.587
5	0.675	0.74	0.697	0.765	0.4
6	0.585	0.615	0.585	0.64	0.343
7	0.56	0.64	0.608	0.65	0.217
8	0.535	0.56	0.552	0.605	0.213
9	0.5	0.55	0.522	0.58	0.075
10	0.505	0.61	0.484	0.585	0.025
11	0.42	0.515	0.402	0.51	0
12	0.39	0.515	0.387	0.48	0
13	0.37	0.44	0.35	0.415	0.015
14	0.14	0.19	0.085	0.13	0
15	0.085	0.125	0.015	0.07	0.005
16	0.01	0.035	0	0.016	0.01
17	0.005	0.021	0.005	0.005	0.016""",
"work/scene1_accuracy.pdf"
)


def st_peters():

    convert_csv(
        #"Dense Affnet vs. MonoDepth+AffNet on scene 1 of the Strong ViewPoint Changes Dataset",
        "Accuracy on St. Peter's Square",
"""Acc 5	DenseAffNet	DepthAffNet
7	0.778	0.774
6	0.8	0.796
5	0.854	0.848
4	0.905	0.898
3	0.937	0.934
2	0.934	0.931
1	0.939	0.97
0	1	1""",
"work/st_peters_accuracy.pdf",
difficulties=range(8), y_ticks=[float(i) / 10 for i in range(7, 11)])


if __name__ == '__main__':
    # EVD()
    # example_stacked_bars()
    # bar_plot_example()
    # graph_grid()
    #scene1()
    st_peters()