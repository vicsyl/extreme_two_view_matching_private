import matplotlib.pyplot as plt
from graphs import is_numeric_my


def convert_csv(title, csv_in_str, file_name,
                difficulties=range(18),
                y_ticks=[float(i)/10 for i in range(11)],
                xlabel="Difficulty",
                eps_on_top=0,
                revertx=False,
                vertical_bars=None,
                ):

    legend_fontsize = 'x-large'
    axes_fontsize = 'xx-large'


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
                acc_data_lists_x[i].append(float(list(difficulties)[diff]))
                acc_data_lists_y[i].append(float(t.strip()))

    fig, ax = plt.subplots(1, figsize=(8, 6))

    if vertical_bars is not None:
        min_y_t = min(y_ticks)
        m = max(vertical_bars)
        vertical_bars = [v * (1 - min_y_t) / m + min_y_t for v in vertical_bars]
        width = 0.05
        ax.bar(acc_data_lists_x[0], vertical_bars, align='center', width=width)

    for i, _ in enumerate(acc_data_lists_x):
        ax.plot(acc_data_lists_x[i], acc_data_lists_y[i], styles[i % len(styles)], label=entries[i])

    if revertx:
        plt.xlim([difficulties[-1], difficulties[0]])
    else:
        plt.xlim([difficulties[0], difficulties[-1]])

    plt.xticks(difficulties)
    plt.ylim(min(y_ticks), max(y_ticks) + eps_on_top)
    plt.yticks(y_ticks)

    plt.legend(shadow=True, framealpha=None, fontsize=legend_fontsize)

    plt.xlabel(xlabel, fontsize=axes_fontsize)
    plt.ylabel("Accuracy (relative rotation error < 5°)", fontsize=axes_fontsize)

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=False, which='minor', color='#999999', linestyle='-', alpha=0.2, axis='y')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
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


def scene1():

    convert_csv(
        "Accuracy on scene 1",
"""\tDenseAffNet\tDepthAffNet\tsimple depth-based\tsimple AffNet shapes\tunrectified(SIFT+HardNet)
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


# def st_peters_cdf():
#
#     convert_csv(
#         "Accuracy on St. Peter's Square",
# """Acc 5	DenseAffNet	DepthAffNet
# 7	0.778	0.774
# 6	0.8	0.796
# 5	0.854	0.848
# 4	0.905	0.898
# 3	0.937	0.934
# 2	0.934	0.931
# 1	0.939	0.97
# 0	1	1""",
# "work/st_peters_accuracy.pdf",
# difficulties=range(7, -1, -1),
#         y_ticks=[float(i) / 10 for i in range(7, 11)],
#         xlabel="Category",
#         eps_on_top=0.01,
#         revertx=True,
#     )
#
#
def st_peters_pdf():

    convert_csv(
        "Accuracy on St. Peter's Square",
"""Acc 5	DenseAffNet	DepthAffNet
7	0.3859315589	0.3819315589
6	0.6625491679	0.6636399395
5	0.7879365621	0.7832319236
4	0.8796716981	0.8695056604
3	0.9385846995	0.9355846995
2	0.9325267857	0.9195089286
1	0.9317627119	0.966440678
0	1	1""",
"work/st_peters_accuracy_pdf.pdf",
difficulties=[i / 10 for i in range(1, 9)],
        y_ticks=[float(i) / 10 for i in range(3, 11)],
        xlabel="Category",
        eps_on_top=0.01,
        revertx=True,
        vertical_bars = [263, 1322, 1466, 1060, 549, 224, 59, 7]
    )


def sacre_coeur_pdf():

    convert_csv(
        "Accuracy on Sacre Coeur",
"""Counts	DenseAffNet	DepthAffNet
0	0.5618181818	0.5695
1	0.8559359561	0.8483659652
2	0.9448722678	0.9142643443
3	0.9539279701	0.9153554724
4	0.964	0.9493333333
5	0.9802941176	0.9701764706
6	0.9705242718	0.980776699
7	1	1
8	1	1""",
"work/sacre_coeur_pdf.pdf",
difficulties=[i / 10 for i in range(1, 10)],
        y_ticks=[float(i) / 10 for i in range(5, 11)],
        xlabel="Category",
        eps_on_top=0.01,
        revertx=True,
        vertical_bars=column_str_to_list("""484
                                            1093
                                            1464
                                            1069
                                            504
                                            204
                                            103
                                            24
                                            5"""),
    )


def reichstag_pdf():

    convert_csv(
        "Accuracy on Reichstag",
"""	DenseAffNet	DepthAffNet
7	0.6376394558	0.6667619048
6	0.8842752294	0.8535806029
5	0.9649294118	0.9017176471
4	0.986835443	0.985443038
3	0.98378125	0.984875
2	0.9565217391	1
1	1	1
0	1	1""",
"work/reichstag_pdf.pdf",
difficulties=[i / 10 for i in range(1, 9)],
        y_ticks=[float(i) / 10 for i in range(6, 11)],
        xlabel="Category",
        eps_on_top=0.01,
        revertx=True,
        vertical_bars=column_str_to_list("""147
                                            763
                                            680
                                            869
                                            192
                                            23
                                            18
                                            9"""),
    )


def column_str_to_list(s):
    return [int(token.strip()) for token in s.split("\n") if token.strip() != ""]


if __name__ == '__main__':
    # EVD()
    # example_stacked_bars()
    # bar_plot_example()
    # graph_grid()
    scene1()
    st_peters_pdf()
    sacre_coeur_pdf()
    reichstag_pdf()