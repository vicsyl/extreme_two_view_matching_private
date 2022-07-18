import matplotlib.pyplot as plt
from graphs import is_numeric_my


def convert_csv(title, csv_in_str, file_name,
                difficulties=range(18),
                y_ticks=[float(i)/10 for i in range(11)],
                xlabel="Relative pose angular difference threshold [deg.]",
                eps_on_top=0,
                revertx=False,
                vertical_bars=None,
                styles=None,
                location=None,
                markersize=5,
                custom_xticks=None,
                add_img=False,
                ):

    legend_fontsize = 'x-large'
    axes_fontsize = 'xx-large'

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot:~:text=Notes-,Format,-Strings
    # markers: . , o v ^ > < s p * + x d | _
    # line styles: - -- -. :
    # fmt=[marker][color][line]
    if styles is None:
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
        ax.bar(acc_data_lists_x[0], vertical_bars, align='center', width=width, color="skyblue")

    for i, _ in enumerate(acc_data_lists_x):
        ax.plot(acc_data_lists_x[i], acc_data_lists_y[i], styles[i % len(styles)], label=entries[i],  linewidth=1.5, markersize=markersize)

    if revertx:
        plt.xlim([difficulties[-1], difficulties[0]])
    else:
        plt.xlim([difficulties[0], difficulties[-1]])

    if custom_xticks is None:
        plt.xticks(difficulties)
    else:
        plt.xticks(difficulties, custom_xticks)
    plt.ylim(min(y_ticks), max(y_ticks) + eps_on_top)
    plt.yticks(y_ticks)


    if location is None:
        location = 'best'
    plt.legend(shadow=True, framealpha=None, fontsize=legend_fontsize, loc=location)

    plt.xlabel(xlabel, fontsize=axes_fontsize)
    plt.ylabel("Accuracy (relative rotation error < 5°)", fontsize=axes_fontsize)

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=False, which='minor', color='#999999', linestyle='-', alpha=0.2, axis='y')

    if add_img:
        img = plt.imread("original_dataset/scene1/images/frame_0000000025_4.jpg")
        axx = fig.add_axes([-0.05, 0.15, 0.35, 0.35], anchor='NE', zorder=1)
        axx.set_axis_off()
        axx.imshow(img)

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
"""	DenseAffNet	DepthAffNet	simple depth-map-based	simple AffNetShapes	unrectified AffNet
0	0.95	0.945	0.923	0.935	0.93
1	0.925	0.94	0.903	0.91	0.895
2	0.89	0.91	0.892	0.89	0.885
3	0.87	0.92	0.817	0.905	0.87
4	0.85	0.88	0.766	0.895	0.815
5	0.675	0.74	0.697	0.765	0.63
6	0.585	0.615	0.585	0.64	0.58
7	0.56	0.64	0.608	0.65	0.525
8	0.535	0.56	0.552	0.605	0.45
9	0.5	0.55	0.522	0.58	0.45
10	0.505	0.61	0.484	0.585	0.435
11	0.42	0.515	0.402	0.51	0.295
12	0.39	0.515	0.387	0.48	0.185
13	0.37	0.44	0.35	0.415	0.11
14	0.14	0.19	0.085	0.13	0.01
15	0.085	0.125	0.015	0.07	0.04
16	0.01	0.035	0	0.016	0.005
17	0.005	0.021	0.005	0.005	0""",
"work/scene1_accuracy.pdf",
        styles=scenes_styles,
        markersize=scene_markersize,
        custom_xticks = ["{}°".format(i * 10) for i in range(18)],
        add_img=True,
)


def scenes2_8():

    convert_csv(
        "Accuracy on scene 2 to scene 8",
"""	DenseAffNet	DepthAffNet	simple depth-map-based	simple AffNetShapes 	unrectified AffNet
0	0.9422	0.9557142857	0.93	0.9528571429	0.94
1	0.8607142857	0.8607142857	0.8192857143	0.8628571429	0.8464285714
2	0.7821507143	0.7892721429	0.7193057143	0.8064285714	0.7671207143
3	0.717815	0.7106828571	0.6236085714	0.7285714286	0.6992742857
4	0.63215	0.6306514286	0.5527985714	0.6492878571	0.6142257143
5	0.5656942857	0.5499321429	0.484265	0.5778342857	0.5250085714
6	0.4720935714	0.4721035714	0.4199957143	0.4950528571	0.4607464286
7	0.4377314286	0.4243492857	0.3621364286	0.4514757143	0.4037028571
8	0.3915	0.39153	0.3085721429	0.4093471429	0.3477385714
9	0.3385592857	0.3428457143	0.2499521429	0.3642857143	0.3150257143
10	0.3028942857	0.3193135714	0.2020464286	0.3243242857	0.2620771429
11	0.2756664286	0.2971635714	0.1828242857	0.3150528571	0.2220457143
12	0.2313678571	0.2764114286	0.1300192857	0.2878571429	0.1606607143
13	0.122879538	0.1379051155	0.05377722772	0.1559405941	0.09495049505
14	0.1153560682	0.1251504514	0.05417151454	0.1504262788	0.1043530592
15	0.09645970488	0.0896708286	0.02950056754	0.09421112372	0.07264472191
16	0.05226407507	0.04951608579	0.03488069705	0.04825737265	0.04421447721
17	0.0159	0.009090909091	0.009090909091	0.01819090909	0.009090909091""",
"work/scene2_8_accuracy.pdf",
        styles=scenes_styles,
        markersize=scene_markersize,
        custom_xticks=["{}°".format(i * 10) for i in range(18)],
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

phototourism_styles = [
        "bo-", "gx-", "kP-", "rd-", "y*-"
        #"bo-", "ro+--", "gx-.", "c*-", "ko-"
    ]

scenes_styles = [
        "bo-", "gx-", "kP-", "rd-", "y*-"
        #"bo-", "go-", "ko-", "ro-", "yo-"
        #"bo-", "ro+--", "gx-.", "c*-", "ko-"
    ]

scene_markersize=5
phototourism_markersize=10
phototourism_xlabel="Covisibility threshold"


def st_peters_pdf():

    convert_csv(
        "Accuracy on St. Peter's Square",
"""Acc 5	DenseAffNet	DepthAffNet	simple depth-map-based	simple AffNetShapes	unrectified AffNet
7	0.3859315589	0.3819315589	0.3571102662	0.3154676806	0.3671102662
6	0.6625491679	0.6636399395	0.6550945537	0.658730711	0.6701853253
5	0.7879365621	0.7832319236	0.7756412005	0.7582783083	0.7862319236
4	0.8796716981	0.8695056604	0.8704632075	0.8690886792	0.8748801887
3	0.9385846995	0.9355846995	0.9249435337	0.9347540984	0.9303023679
2	0.9325267857	0.9195089286	0.9247589286	0.9156964286	0.9324553571
1	0.9317627119	0.966440678	0.9317627119	0.9317627119	0.966440678
0	1	1	1	1	1""",
"work/st_peters_accuracy_pdf.pdf",
difficulties=[i / 10 for i in range(1, 9)],
        y_ticks=[float(i) / 10 for i in range(3, 11)],
        xlabel=phototourism_xlabel,
        eps_on_top=0.01,
        revertx=True,
        vertical_bars = [263, 1322, 1466, 1060, 549, 224, 59, 7],
        styles=phototourism_styles,
        location=(0.01, 0.4),
        markersize=phototourism_markersize,
    )


def sacre_coeur_pdf():

    convert_csv(
        "Accuracy on Sacre Coeur",
"""Counts	DenseAffNet	DepthAffNet	simple depth-map-based	simple AffNetShapes	unrectified AffNet
4950	0.5618181818	0.5695	0.5607272727	0.5485909091	0.6987727273
4466	0.8559359561	0.8483659652	0.8324519671	0.8560219579	0.8877099726
3373	0.9448722678	0.9142643443	0.8942643443	0.9408722678	0.9411762295
1909	0.9539279701	0.9153554724	0.8992843779	0.9514995323	0.9479279701
840	0.964	0.9493333333	0.933	0.956	0.954
336	0.9802941176	0.9701764706	0.9418823529	0.9701764706	0.9802941176
132	0.9705242718	0.980776699	0.9224271845	0.980776699	0.9705242718
29	1	1	0.9589166667	1	1
5	1	1	1	1	1""",
"work/sacre_coeur_pdf.pdf",
difficulties=[i / 10 for i in range(1, 10)],
        y_ticks=[float(i) / 10 for i in range(5, 11)],
        xlabel=phototourism_xlabel,
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
        styles=phototourism_styles,
        location=(0.01, 0.4),
        markersize=phototourism_markersize,
    )


def reichstag_pdf():

    convert_csv(
        "Accuracy on Reichstag",
"""	DenseAffNet	DepthAffNet	simple depth-map-based	simple AffNetShapes	unrectified AffNet
7	0.6376394558	0.6667619048	0.6250136054	0.5845170068	0.6386394558
6	0.8842752294	0.8535806029	0.8438859764	0.8815806029	0.8852752294
5	0.9649294118	0.9017176471	0.8987176471	0.9689294118	0.9659294118
4	0.986835443	0.985443038	0.9816075949	0.990556962	0.9881139241
3	0.98378125	0.984875	0.9900833333	0.9900833333	0.98378125
2	0.9565217391	1	0.9565217391	0.9565217391	0.9565217391
1	1	1	1	1	1
0	1	1	1	1	1""",
"work/reichstag_pdf.pdf",
difficulties=[i / 10 for i in range(1, 9)],
        y_ticks=[float(i) / 10 for i in range(6, 11)],
        xlabel=phototourism_xlabel,
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
        styles=phototourism_styles,
        location=(0.01, 0.4),
        markersize=phototourism_markersize,
    )


def column_str_to_list(s):
    return [int(token.strip()) for token in s.split("\n") if token.strip() != ""]


if __name__ == '__main__':
    # EVD()
    # example_stacked_bars()
    # bar_plot_example()
    # graph_grid()

    scene1()
    scenes2_8()
    st_peters_pdf()
    sacre_coeur_pdf()
    reichstag_pdf()