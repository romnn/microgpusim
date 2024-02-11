def find_cache_set_mapping():
    # if compute_capability is None:
    #     compute_capability = remote.get_compute_capability(gpu=gpu)

    # if compute_capability == 86:
    #     if average is None:
    #         average = False

    # assert (
    #     known_cache_size_bytes
    #     == known_num_sets * derived_num_ways * known_cache_line_bytes
    # )
    # if gpu == "A4000":
    #     # stride_bytes = 32
    #     pass

    # # manual averaging based on the pattern
    # def most_common_pattern(_df):
    #     print(_df)
    #     print(_df.shape)
    #     return _df
    #
    # def get_misses(_df):
    #     # print(_df)
    #     return _df.loc[_df["hit_cluster"] != 0, "index"]

    # assert len(unique_indices) == known_cache_size_bytes / stride_bytes

    # if compute_capability == 86:
    #     if not random:
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             # assert len(_df["index"].unique().tolist()) == len(unique_indices)
    #             mask = combined["n"] == n
    #             mask &= combined["overflow_index"] == overflow_index
    #             mask &= combined["r"] == r
    #             stride = stride_bytes / 4
    #             wrapped = combined["index"] <= overflow_index - stride_bytes
    #             combined.loc[mask & wrapped, "latency"] = 300.0
    #
    #     if random:
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             rounds_1_and_2 = _df["k"] >= 1 * round_size
    #             rounds_1_and_2 &= _df["k"] < 3 * round_size
    #             # print(_df[rounds_1_and_2].head(n=150))
    #             # print(_df.loc[rounds_1_and_2, "hit_cluster"] > 0)
    #             num_misses = (_df.loc[rounds_1_and_2, "hit_cluster"] > 0).sum()
    #             if num_misses == 0:
    #                 print(color("n={} overflow index={} num misses={}".format(
    #                     n, overflow_index, num_misses), fg="yellow"))
    #             elif num_misses != 1:
    #                 raise ValueError("more than one miss in round 1 and 2")
    #
    #
    #     if False:
    #         patterns = dict()
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             if (n, overflow_index) not in patterns:
    #                 patterns[(n, overflow_index)] = defaultdict(int)
    #
    #             misses = tuple(_df.loc[_df["hit_cluster"] != 0, "index"].tolist())
    #             # print(misses)
    #             patterns[(n, overflow_index)][misses] += 1
    #
    #         # pprint(patterns)
    #         # get most common patterns
    #         most_common_patterns = {
    #             k: sorted(v.items(), key=lambda x: x[1])[-1] for k, v in patterns.items()}
    #         # pprint(most_common_patterns)
    #
    #         new_combined = []
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             misses = tuple(_df.loc[_df["hit_cluster"] != 0, "index"].tolist())
    #             if most_common_patterns[(n, overflow_index)][0] == misses:
    #                 # print("adding", (n, overflow_index, r))
    #                 mask = combined["n"] == n
    #                 mask &= combined["overflow_index"] == overflow_index
    #                 mask &= combined["r"] == r
    #                 assert len(combined[mask]) == len(unique_ks)
    #                 new_combined.append(combined[mask])
    #                 most_common_patterns[(n, overflow_index)] = (None, None)
    #
    #         combined = pd.concat(new_combined, ignore_index=True)
    #         assert (
    #             len(combined[["n", "overflow_index"]].drop_duplicates())
    #             == len(combined[["n", "overflow_index", "r"]].drop_duplicates()))
    #
    #         combined["r"] = 0
    #         repetitions = 1

    # combined = combined.sort_values(["n", "overflow_index", "k", "r"])

    # print(col_miss_count)

    # all_hits = np.all(hit_clusters == 0, axis=0)
    # (all_hits_indices,) = np.where(all_hits == True)
    # print(all_hits_indices)

    # shifted plot
    # if False:
    #     num_overflow_indices = len(combined["overflow_index"].unique())
    #     # num_overflow_indices = int(known_cache_size_bytes / stride_bytes)
    #     # print(num_overflow_indices)
    #     # assert(num_overflow_indices == len(combined["overflow_index"].unique()))
    #     # assert(num_overflow_indices == int(max_rows / repetitions))
    #     all_latencies = latencies.copy()
    #     all_latencies[:,round_size:] = 0.0
    #     shift = "left"
    #     for overflow_index in range(num_overflow_indices):
    #         row_start = overflow_index * repetitions
    #         row_end = (overflow_index+1)*repetitions
    #         if shift == "right":
    #             new_col_start = round_size + overflow_index
    #             new_col_end = max_cols
    #             col_start = round_size
    #             col_end = min(max_cols, col_start + (new_col_end - new_col_start))
    #         elif shift == "left":
    #             pass
    #             col_start = round_size + round_size
    #             col_end = max_cols
    #             new_col_start = round_size
    #             new_col_end = min(max_cols, new_col_start + (col_end - col_start))
    #         else:
    #             raise ValueError
    #         all_latencies[row_start:row_end,new_col_start:new_col_end] = latencies[row_start:row_end,col_start:col_end]
    #
    #     fig = plot_access_process_latencies(
    #         combined,
    #         all_latencies,
    #         warmup=warmup,
    #         rounds=max_rounds,
    #         ylabel="overflow index",
    #         size_bytes=known_cache_size_bytes,
    #         stride_bytes=stride_bytes,
    #     )
    #
    #     filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    #     filename = filename.with_stem(filename.name + "_shifted")
    #     filename.parent.mkdir(parents=True, exist_ok=True)
    #     print("saved plot to {}".format(filename))
    #     fig.savefig(filename)

    # per overflow plot
    if False:
        for (overflow_addr_index,), _ in combined.groupby(["overflow_index"]):
            stride = float(stride_bytes) / 4.0
            overflow_index = float(overflow_addr_index) / stride
            overflow_latencies = latencies[
                int(overflow_index * repetitions) : int(
                    (overflow_index + 1) * repetitions
                ),
                :,
            ]
            fig = plot_access_process_latencies(
                combined,
                overflow_latencies,
                warmup=warmup,
                rounds=max_rounds,
                ylabel="repetition",
                size_bytes=known_cache_size_bytes,
                stride_bytes=stride_bytes,
            )

            filename = PLOT_DIR / "pchase_overflow" / cache_file.relative_to(CACHE_DIR)
            filename = filename.with_name(
                "overflow_{}".format(int(overflow_index))
            ).with_suffix(".pdf")
            filename.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filename)
            print("saved to ", filename)

    if False:
        # group by first miss in round 1 and 2
        print(
            combined.loc[
                (combined["k"] >= 1 * round_size) & (combined["k"] < 3 * round_size), :
            ].head(n=100)
        )

        # print(stride_bytes / 4)
        # total_sets = {k * 4: list() for k in range(1 * round_size, 3 * round_size, int(stride_bytes / 4))}
        total_sets = {
            int(k): list() for k in range(0, known_cache_size_bytes, stride_bytes)
        }
        # print(total_sets)
        # before = len(total_sets)
        if random:
            for (n, overflow_index, r), _df in combined.groupby(
                ["n", "overflow_index", "r"]
            ):
                rounds_3 = _df["k"] >= 3 * round_size

                rounds_1_and_2 = _df["k"] >= 1 * round_size
                rounds_1_and_2 &= _df["k"] < 3 * round_size
                first_miss = _df.loc[rounds_1_and_2 & (_df["hit_cluster"] > 0), "index"]
                assert len(first_miss) <= 1
                if len(first_miss) > 0:
                    first_miss = int(first_miss.iloc[0])
                    misses = tuple(
                        _df.loc[rounds_3 & (_df["hit_cluster"] > 0), "index"].tolist()
                    )
                    total_sets[first_miss].append(misses)

        # pprint(total_sets)
        # assert before == len(total_sets)

        total_sets_df = pd.DataFrame.from_records(
            [(k, len(v)) for k, v in total_sets.items()], columns=["index", "count"]
        )
        print(total_sets_df)

        fig = plt.figure(
            figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
            layout="constrained",
        )
        ax = plt.axes()
        ax.scatter(total_sets_df["index"], total_sets_df["count"], 10, marker="o")
        filename = PLOT_DIR / "set_freqs.pdf"
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename)
        print("saved to ", filename)

        total_sets_df["prob"] = total_sets_df["count"] / len(total_sets_df)
        print(total_sets_df)

    # # print(combined["latency"].value_counts())
    # # print(combined["hit_cluster"].unique())
    # # min_latency = combined["latency"].min()
    # # max_latency = combined["latency"].max()
    #
    # mean_cluster_latency = combined.groupby("hit_cluster")["latency"].mean()
    # min_cluster_latency = combined.groupby("hit_cluster")["latency"].min()
    # max_cluster_latency = combined.groupby("hit_cluster")["latency"].max()
    # max_latency = mean_cluster_latency[1] + 100
    # min_latency = np.max([0, mean_cluster_latency[0] - 100])
    #
    # ylabel = r"overflow index"
    # xlabel = r"cache access process"
    # fontsize = plot.FONT_SIZE_PT
    # font_family = "Helvetica"
    #
    # plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
    #
    # fig = plt.figure(
    #     figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
    #     layout="constrained",
    # )
    # ax = plt.axes()
    #
    # white = (255,255,255)
    # orange = (255,140,0)
    # red = (255,0,0)
    #
    # def rgb_to_vec(color: typing.Tuple[int, int, int], alpha:float = 1.0):
    #     return np.array([color[0]/255.0, color[1]/255.0, color[2]/255.0, alpha])
    #
    # def gradient(start: typing.Tuple[int, int, int], end: typing.Tuple[int, int, int], n=256):
    #     vals = np.ones((n, 4))
    #     vals[:, 0] = np.linspace(start[0]/255.0, end[0]/255.0, n)
    #     vals[:, 1] = np.linspace(start[1]/255.0, end[1]/255.0, n)
    #     vals[:, 2] = np.linspace(start[2]/255.0, end[2]/255.0,  n)
    #     return ListedColormap(vals)
    #
    # white_to_orange = gradient(start=white, end=orange)
    # orange_to_red = gradient(start=orange, end=red)
    #
    # latency_range = max_latency - min_latency
    # mean_hit_cluster_latency = mean_cluster_latency[0]
    # mean_miss_cluster_latency = mean_cluster_latency[1]
    # assert min_latency <= mean_hit_cluster_latency <= mean_miss_cluster_latency <= max_latency
    #
    # tol = 0.2
    # start = min_latency
    # hit_end = min_cluster_latency[1] - tol * abs(min_cluster_latency[1] - max_cluster_latency[0])
    # # hit_end = mean_hit_cluster_latency + tol * (mean_miss_cluster_latency - mean_hit_cluster_latency)
    # # miss_end = mean_miss_cluster_latency + tol * (max_latency - mean_miss_cluster_latency)
    # miss_end = np.min([mean_cluster_latency[1] + 100, max_latency])
    # end = miss_end
    #
    # points = [start, hit_end, miss_end, end]
    # widths = [points[i+1] - points[i] for i in range(len(points) - 1)]
    #
    # assert np.sum(widths) == latency_range
    #
    # latency_cmap = np.vstack([
    #     np.repeat(rgb_to_vec(white).reshape(1, 4), repeats=int(np.round(widths[0])), axis=0),
    #     white_to_orange(np.linspace(0, 1, int(np.round(0.5 * widths[1])))),
    #     orange_to_red(np.linspace(0, 1, int(np.round(0.5 * widths[1])))),
    #     np.repeat(rgb_to_vec(red).reshape(1, 4), repeats=int(np.round(widths[2])), axis=0),
    # ])
    #
    # assert np.allclose(len(latency_cmap), int(np.round(latency_range)), atol=2)
    # latency_cmap = ListedColormap(latency_cmap)
    #
    # if False:
    #     # c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    #     c = ax.pcolormesh(latencies, cmap='RdBu')
    #     # ax.set_title('pcolormesh')
    #     # set the limits of the plot to the limits of the data
    #     # ax.axis([x.min(), x.max(), y.min(), y.max()])
    #     fig.colorbar(c, ax=ax)
    # else:
    #     c = plt.imshow(latencies, cmap=latency_cmap, vmin=min_latency, vmax=max_latency,
    #                    interpolation='nearest',
    #                    origin='upper',
    #                    aspect='auto',
    #                    # aspect="equal",
    #     )
    #     fig.colorbar(c, ax=ax)
    #
    # if False:
    #     # plot all hit locations
    #     print(all_hits_indices)
    #     print(len(all_hits_indices))
    #     for hit_index in all_hits_indices:
    #         ax.axvline(
    #             x=hit_index,
    #             color="black", # plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
    #             linestyle="-",
    #             linewidth=1,
    #             # label=r"L1 hit",
    #         )
    #
    # ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)
    # # xticks = np.arange(min_x, max_x, step=256 * KB)
    # # xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    # # ax.set_xticks(xticks, xticklabels, rotation=45)
    # # ax.set_xlim(min_x, max_x)
    # # ax.set_ylim(0, 100.0)
    # # ax.legend()

    # data = np.array([
    #     np.pad(s, pad_width=(largest_set - len(s), 0), mode='constant', constant_values=0)
    #     for s, _ in total_sets
    # ])
    # print(data.shape)
    #
    # diffs = []
    # for s1, _ in total_sets:
    #     for s2, _ in total_sets:
    #         union = np.union1d(s1, s2)
    #         intersection = np.intersect1d(s1, s2)
    #         diffs.append(len(union) - len(intersection))
    #
    # diffs = np.array(diffs)
    # print("pairwise diff: min={} max={} mean={}".format(np.amin(diffs), np.amax(diffs), np.mean(diffs)))

    # line table
    # print(combined["cache_line"].unique())
    # print(combined["set"].unique())
    # return

    # cache_line_set_mapping = pd.DataFrame(
    #     np.array(combined["cache_line"].unique()), columns=["cache_line"]
    # )
    # # cache_line_set_mapping = combined["cache_line"].unique().to_frame()
    # # print(cache_line_set_mapping)
    # cache_line_set_mapping["mapped_set"] = np.nan
    # cache_line_set_mapping = cache_line_set_mapping.sort_values("cache_line")
    # cache_line_set_mapping = cache_line_set_mapping.reset_index()

    # cache_lines //= known_cache_line_bytes / sector_size_bytes
    # cache_lines = cache_lines[cache_lines < derived_num_ways]
    # cache_lines = sorted(cache_lines.unique().tolist())

    # for line in cache_lines.astype(int):
    #     # way = ((line -1) // 4) % derived_num_ways
    #     way = line % derived_num_ways
    #     # print(way, int(mapped_set))
    #     # line_table.iloc[way, int(mapped_set) - 1] = line
    # end of line table

    def custom_dist(a, b):
        # assert list(a) == sorted(list(a))
        # assert list(b) == sorted(list(b))
        # a = np.unique(a)
        # b = np.unique(b)
        # print("a", a.shape)
        # print("b", b.shape)
        # assert a.shape == b.shape
        union = np.union1d(a, b)
        intersection = np.intersect1d(a, b)
        # print("union", union.shape)
        # print("intersection", intersection.shape)

        union = len(union[union != 0])
        intersection = len(intersection[intersection != 0])
        if union == 0:
            union_intersect = 0
        else:
            union_intersect = intersection / union
        # print("match = {:<4.2f}%".format(union_intersect * 100.0))
        diff = len(a[a != 0]) - intersection
        # print(diff)
        # return diff
        return union_intersect

    cluster_labels = DBSCAN(
        # eps=0.05, # 5 percent difference
        eps=0.001,  # 5 percent difference
        min_samples=1,
        metric=custom_dist,
        # metric_params={'w1':1,'w2':5,'w3':4},
    ).fit_predict(data)
    # ).fit_predict(data.reshape(-1, 1))
    print(cluster_labels)

    if gpu == "A4000":
        print("todo")
        return

    # def check_duplicates(needle):
    #     count = 0
    #     for s in total_sets:
    #         for addr in s:
    #             if addr - base_addr == needle:
    #                 count += 1
    #         if count > 1:
    #             break
    #     if count > 1:
    #         return str(color(str(needle), fg="red"))
    #     return str(needle)

    # already been done
    # for set_id, s in enumerate(total_sets):
    #     print(
    #         "set {: <4}\t [{}]".format(
    #             set_id, ", ".join([check_duplicates(addr - base_addr) for addr in s])
    #         )
    #     )
    #     combined.loc[combined["virt_addr"].astype(int).isin(s), "set"] = set_id

    # set_boundary_addresses = range(0, known_cache_size_bytes, known_cache_line_bytes)
    # assert len(set_boundary_addresses) == len(set_bits)
    # for index, offset in zip(set_boundary_addresses, set_bits):

    # def get_set_bit_mapping(bit: int):
    #     return [
    #         (
    #             int(addr),
    #             bitarray.util.int2ba(
    #                 int(set_id), length=num_sets_log2, endian="little"
    #             )[bit],
    #         )
    #         for addr, set_id in combined[["virt_addr", "set"]].to_numpy()
    #     ]

    # manual search for the set mapping function
    for set_bit, f in found_mapping_functions.items():
        # if set_bit == 0:
        #     continue
        print("==== SET BIT {:<2}".format(set_bit))

        bit_mapping = get_offset_bit_mapping(bit=set_bit)
        bit_pos_used = sorted(
            [
                int(str(bit).removeprefix("~").removeprefix("b"))
                for bit in unique_bits(f)
            ]
        )
        bits_used = [sym.symbols(f"b{pos}") for pos in bit_pos_used]

        # bits_used = sorted(
        #     [sym.symbols(bit) for bit in unique_bits(f)],
        #     key=lambda b: int(str(b).removeprefix("~").removeprefix("b")))

        print("unique bits: {}".format(bits_used))
        print_cnf_terms(f)

        f = sym.logic.boolalg.to_dnf(f, simplify=True, force=True)
        print("original", f)

        optimized = True
        while optimized:
            optimized = False
            for bit in bits_used:
                if str(bit) != "b10":
                    continue
                ff = remove_variable(f, var=bit)
                print("remove {:<5} => {}".format(str(bit), ff))

                assert isinstance(ff, sym.Or)
                assert len(ff.args) == len(f.args)

                substitutions = {
                    # "xor_bit": sym.logic.boolalg.Or(*[term for term in ff.args]),
                    # "xor_bit": sym.logic.boolalg.Or(*[sym.logic.boolalg.And(term, sym.symbols(f"b{bit}")) for term in ff.args]),
                    "xor_bit": sym.logic.boolalg.Or(
                        *[
                            (
                                sym.logic.boolalg.Xor(term, bit)
                                if contains_var(f.args[i], var=bit)
                                else term
                            )
                            for i, term in enumerate(ff.args)
                        ]
                    ),
                    # "xor_bit": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, bit) for term in ff.args]),
                    # "xor_not_bit": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit) for term in ff.args]),
                    # "xor_bit_twice": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, bit, bit) for term in ff.args]),
                    # "xor_not_bit_twice": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit, ~bit) for term in ff.args]),
                }
                # substitutions = {
                #     # "remove_bit": ff,
                #     "xor_bit": sym.logic.boolalg.Xor(ff, bit),
                #     # "xor_not_bit": sym.logic.boolalg.Xor(ff, ~bit),
                #     # "xor_bit_twice": sym.logic.boolalg.Xor(ff, bit, bit),
                #     # "xor_not_bit_twice": sym.logic.boolalg.Xor(ff, ~bit, ~bit),
                # }
                for sub_name, sub_f in substitutions.items():
                    print(sub_f)
                    valid = True
                    for addr, target_bit in bit_mapping:
                        index_bits = bitarray.util.int2ba(
                            addr, length=64, endian="little"
                        )
                        vars = {
                            bit: index_bits[pos]
                            for bit, pos in reversed(list(zip(bits_used, bit_pos_used)))
                        }
                        ref_pred = int(bool(f.subs(vars)))
                        sub_pred = int(bool(sub_f.subs(vars)))
                        assert ref_pred == target_bit
                        print(
                            np.binary_repr(addr, width=64),
                            vars,
                            color(
                                sub_pred,
                                fg="red" if sub_pred != target_bit else "green",
                            ),
                            target_bit,
                        )
                        if sub_pred != target_bit:
                            valid = False

                    is_valid_under_simplification = equal_expressions(f, sub_f)
                    assert valid == is_valid_under_simplification
                    print(sym.logic.boolalg.to_dnf(sub_f, simplify=True, force=True))
                    print(sub_name, valid)

                    if valid:
                        f = sub_f
                        optimized = True
                        break

                print("current function {}".format(f))

            # print("final function {}".format(f))
            break

            # ff_xor_bit =
            # print(ff_xor_bit)
            # print(sym.logic.boolalg.to_dnf(ff_xor_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_bit))
            #
            # ff_xor_not_bit = sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit) for term in ff.args])
            # print(sym.logic.boolalg.to_dnf(ff_xor_not_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_not_bit))
            #
            # ff_xor_xor_double_bit = sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, bit, bit) for term in ff.args])
            # print(sym.logic.boolalg.to_dnf(ff_xor_xor_double_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_xor_double_bit))
            #
            # ff_xor_xor_double_not_bit = sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit, ~bit) for term in ff.args])
            # print(sym.logic.boolalg.to_dnf(ff_xor_xor_double_not_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_xor_double_not_bit))

    return

    for set_bit in range(1, num_sets_log2):
        bit_mapping = [
            (
                int(addr),
                bitarray.util.int2ba(
                    int(set_id), length=num_sets_log2, endian="little"
                )[set_bit],
            )
            for addr, set_id in combined[["virt_addr", "set"]].to_numpy()
        ]
        is_offset = True
        bit_mapping = [
            (
                int(addr),
                bitarray.util.int2ba(
                    int(offset), length=num_sets_log2, endian="little"
                )[set_bit],
            )
            for addr, offset in combined[["virt_addr", "offset"]].to_numpy()
        ]

        print("==== SET BIT {:<2}".format(set_bit))
        for addr, target_bit in bit_mapping:
            full_index_bits = bitarray.util.int2ba(
                addr, length=max_bits, endian="little"
            )

            # even xor => 0
            # uneven xor => 1

            vars = {sym.symbols(f"b{b}"): full_index_bits[b] for b in range(max_bits)}
            predicted = found_mapping_functions[set_bit].subs(vars)
            predicted = int(bool(predicted))

            marks = set([])
            # marks = set([3584, 7680, 11776, 19968, 24064])
            # marks = set([1536])

            # CORRECT OFFSET BIT 0
            bit0 = bool(full_index_bits[9])
            bit0 = bit0 ^ bool(full_index_bits[9])
            bit0 = bit0 ^ bool(full_index_bits[10])

            bit0 = bit0 ^ bool(full_index_bits[12])
            bit0 = bit0 ^ bool(full_index_bits[14])
            # if bool(full_index_bits[9]) ^ bool(full_index_bits[10]):
            #     # predicted = not predicted
            #     pass

            if is_offset and set_bit == 0:
                predicted = bit0

            if is_offset and set_bit == 1:
                # CORRECT OFFSET BIT 1
                predicted = bool(full_index_bits[9])
                predicted = predicted ^ (not bit0)
                predicted = predicted ^ bool(full_index_bits[10])
                predicted = predicted ^ bool(full_index_bits[11])
                predicted = predicted ^ bool(full_index_bits[12])
                predicted = predicted ^ bool(full_index_bits[13])
                predicted = predicted ^ bool(full_index_bits[14])

            if not is_offset:
                # PREDICTED 0
                predicted0 = bool(full_index_bits[7])
                # predicted = predicted ^ bool(full_index_bits[9])
                # predicted = predicted ^ (not bool(full_index_bits[9]))
                predicted0 = predicted0 ^ bool(full_index_bits[10])
                # predicted = predicted ^ bool(full_index_bits[11])
                predicted0 = predicted0 ^ bool(full_index_bits[12])
                # predicted = predicted ^ bool(full_index_bits[13])
                predicted0 = predicted0 ^ bool(full_index_bits[14])

                if set_bit == 0:
                    predicted = predicted0

                if set_bit == 1:
                    # this is for the offset only:
                    if False:
                        predicted = bool(full_index_bits[9]) ^ bool(full_index_bits[11])
                        predicted = predicted ^ bool(full_index_bits[13])
                        if False:
                            if full_index_bits[9] & full_index_bits[10]:
                                predicted = bool(not predicted)
                                # print([full_index_bits[b] for b in [11, 12, 13, 14]])
                                # lol = bool(full_index_bits[11]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13]) ^ bool(full_index_bits[14]) ^ bool(full_index_bits[15])
                                # print(predicted, lol)
                                # predicted = predicted ^ lol
                            if (
                                full_index_bits[9]
                                & full_index_bits[10]
                                & full_index_bits[11]
                                & full_index_bits[12]
                            ):
                                predicted = bool(not predicted)

                        if (
                            full_index_bits[9]
                            & full_index_bits[10]
                            & ~(full_index_bits[11] & full_index_bits[12])
                        ):
                            predicted = bool(not predicted)

                    # predicted = bool(full_index_bits[7])
                    predicted = bool(full_index_bits[8])
                    # predicted = predicted ^ bool(full_index_bits[8])
                    # predicted = predicted ^ bool(full_index_bits[7])
                    predicted = predicted ^ (not bool(full_index_bits[7]))

                    predicted = predicted ^ bool(full_index_bits[9])
                    predicted = predicted ^ bool(full_index_bits[10])
                    predicted = predicted ^ bool(full_index_bits[11])
                    # predicted = predicted ^ (not bool(full_index_bits[12]))
                    predicted = predicted ^ bool(full_index_bits[12])
                    predicted = predicted ^ bool(full_index_bits[13])
                    predicted = predicted ^ bool(full_index_bits[14])

                    predicted = not predicted

                    inverter = False

                    section1 = full_index_bits[9:11] == bitarray.bitarray(
                        "11", endian="big"
                    ) or full_index_bits[9:11] == bitarray.bitarray("01", endian="big")
                    section2 = full_index_bits[11:13] != bitarray.bitarray(
                        "11", endian="big"
                    )
                    # section2 = (
                    #     full_index_bits[11:13] == bitarray.bitarray("11", endian="big")
                    #     or full_index_bits[11:13] == bitarray.bitarray("01", endian="big")
                    # )

                    # if (
                    #     full_index_bits[9:11] == bitarray.bitarray("11", endian="big")
                    #     or full_index_bits[9:11] == bitarray.bitarray("01", endian="big")
                    # ):
                    # if section1 and section2:
                    t1 = full_index_bits[10:12] != bitarray.bitarray("01", endian="big")
                    t2 = full_index_bits[10:12] != bitarray.bitarray("11", endian="big")
                    t3 = full_index_bits[10:12] != bitarray.bitarray("00", endian="big")
                    t4 = full_index_bits[10:12] != bitarray.bitarray("10", endian="big")
                    # print(full_index_bits[10:12])

                    # 000
                    # 010
                    # 101
                    # 111
                    t1 = full_index_bits[10:13] == bitarray.bitarray(
                        "000", endian="big"
                    )
                    t2 = full_index_bits[10:13] == bitarray.bitarray(
                        "010", endian="big"
                    )
                    t3 = full_index_bits[10:13] == bitarray.bitarray(
                        "101", endian="big"
                    )
                    t4 = full_index_bits[10:13] == bitarray.bitarray(
                        "111", endian="big"
                    )
                    print(full_index_bits[10:13])

                    t5 = full_index_bits[13:15] != bitarray.bitarray("11", endian="big")
                    # 11
                    # 00
                    # 01
                    if not (t1 | t2 | t3 | t4):
                        # predicted = predicted ^ bool(full_index_bits[7])
                        inverter = True

                    offset = bool(full_index_bits[9]) ^ bool(full_index_bits[11])
                    offset = offset ^ bool(full_index_bits[13])
                    # if full_index_bits[9] & full_index_bits[10] & ~(full_index_bits[11] & full_index_bits[12]):
                    #     predicted = bool(not predicted)

                    # if (bool(full_index_bits[9]) & bool(full_index_bits[10])) | ((not bool(full_index_bits[9])) & bool(full_index_bits[10])):
                    # 9 & 10 & ~(11 & 12)
                    # if bool(full_index_bits[7]):
                    #     predicted = not predicted

                    # mask = bool(full_index_bits[11]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13]) ^ bool(full_index_bits[14]) ^ bool(full_index_bits[7])
                    # if mask:
                    #     print(mask)

                    # sector1 = (not bool(full_index_bits[10])) & (not bool(full_index_bits[11]))
                    # sector2 = bool(full_index_bits[11]) & bool(full_index_bits[12])

                    # have:
                    # 00000
                    # 00010
                    # 00101
                    # 00111
                    # 01000
                    # 01010
                    # 01101
                    # 01111
                    # 10001
                    # 10011
                    # 10100
                    # 10110

                    # 000
                    # 010
                    # 101
                    # 111

                    # 000
                    # 010
                    # 101
                    # 111

                    # 001
                    # 011
                    # 100
                    # 110

                    # not 001
                    # not 011
                    # not 100
                    # not 110

                    # not 000
                    # not 010
                    # not 101
                    # not 111

                    # not 11---

                    # 11000
                    # 11000
                    # 10111
                    # 10111
                    # 10101
                    # 10101
                    # 10010
                    # 10010
                    # 10000
                    # 10000
                    # 01110
                    # 01110
                    # 01100
                    # 01100
                    # 01011
                    # 01011
                    # 00110
                    # 00110
                    # 00100
                    # 00100
                    # 00011
                    # 00011
                    # 00001
                    # 00001

                    # 110000
                    # 110001
                    # 101110
                    # 101111
                    # 101010
                    # 101011
                    # 100100
                    # 100101
                    # 100000
                    # 100001
                    # 011100
                    # 011101
                    # 011000
                    # 011001
                    # 010110
                    # 010111
                    # 001100
                    # 001101
                    # 001000
                    # 001001
                    # 000110
                    # 000111
                    # 000010
                    # 000011
                    # if (not bool(full_index_bits[10])) & (not bool(full_index_bits[13])):
                    #     pass
                    # if sector1 | sector2:
                    # if sector1:
                    #     predicted = not predicted

                    if False:
                        predicted = predicted ^ bool(full_index_bits[10])
                        # predicted = predicted ^ bool(full_index_bits[10])
                        predicted = predicted ^ bool(full_index_bits[11])
                        predicted = predicted ^ bool(full_index_bits[12])
                        predicted = predicted ^ bool(full_index_bits[13])
                        predicted = predicted ^ bool(full_index_bits[14])
                        # predicted = predicted ^ bool(full_index_bits[15])
                        # predicted = predicted ^ bool(full_index_bits[11])

                    # special = bool(full_index_bits[11]) ^ bool(full_index_bits[13])
                    # if special:
                    #     # predicted = bool(~predicted)
                    #     pass
                    # ^ full_index_bits[14]
                    # predicted |= full_index_bits[10] ^ full_index_bits[12] ^ full_index_bits[14]

                    # predicted = bool(not predicted)
                    # predicted &= ~(full_index_bits[9] & full_index_bits[10])
                    # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                    # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                    # predicted &= special_case
                    # special_case2 = ~(full_index_bits[10] & full_index_bits[11])
                    # predicted |= full_index_bits[10] ^ full_index_bits[11]

            print(
                "{}\t\t{} => {:>2} {:>2} {} \t bit0={:<1} inverter={:<1}".format(
                    addr,
                    "|".join(
                        split_at_indices(
                            np.binary_repr(
                                bitarray.util.ba2int(full_index_bits), width=num_bits
                            ),
                            indices=[
                                0,
                                num_bits - 15,
                                num_bits - line_size_log2 - num_sets_log2,
                                num_bits - line_size_log2,
                            ],
                        )
                    ),
                    target_bit,
                    str(
                        color(
                            int(predicted),
                            fg=(
                                "green"
                                if bool(predicted) == bool(target_bit)
                                else "red"
                            ),
                        )
                    ),
                    str(color("<==", fg="blue")) if addr in marks else "",
                    str(0),
                    str(0),
                    # predicted0,
                    # str(color(str(int(inverter)), fg="cyan")) if inverter else str(int(inverter)),
                )
            )

    return

    offset_bit_0 = [int(np.binary_repr(o, width=2)[0]) for o in offsets]
    offset_bit_1 = [int(np.binary_repr(o, width=2)[1]) for o in offsets]

    if False:
        print(offset_bit_0)
        print(offset_bit_1)

        for name, values in [
            ("offset bit 0", offset_bit_0),
            ("offset bit 1", offset_bit_1),
        ]:
            patterns = find_pattern(values=values, num_sets=num_sets)
            if len(patterns) < 0:
                print("NO pattern found for {:<10}".format(name))
            for pattern_start, pattern in patterns:
                print(
                    "found pattern for {:<10} (start={: <2} length={: <4}): {}".format(
                        name, pattern_start, len(pattern), pattern
                    )
                )

        print(
            len(
                list(
                    range(0, known_cache_size_bytes, num_sets * known_cache_line_bytes)
                )
            )
        )
        print(len(offsets))
        assert len(
            list(range(0, known_cache_size_bytes, num_sets * known_cache_line_bytes))
        ) == len(offsets)

    if True:
        found_mapping_functions = dict()
        for set_bit in range(num_sets_log2):
            offset_bits = [
                bitarray.util.int2ba(int(o), length=num_sets_log2, endian="little")[
                    set_bit
                ]
                for o in offsets
            ]
            for min_bits in range(1, num_bits):
                bits_used = [
                    line_size_log2 + num_sets_log2 + bit for bit in range(min_bits)
                ]

                print("testing bits {:<30}".format(str(bits_used)))
                t = logicmin.TT(min_bits, 1)
                validation_table = []

                way_boundary_addresses = range(
                    0, known_cache_size_bytes, num_sets * known_cache_line_bytes
                )
                for index, offset in zip(way_boundary_addresses, offset_bits):
                    index_bits = bitarray.util.int2ba(
                        index, length=max_bits, endian="little"
                    )
                    new_index_bits = bitarray.bitarray(
                        [index_bits[b] for b in reversed(bits_used)]
                    )
                    new_index = bitarray.util.ba2int(new_index_bits)
                    new_index_bits_str = np.binary_repr(new_index, width=min_bits)
                    t.add(new_index_bits_str, str(offset))
                    validation_table.append((index_bits, offset))

                sols = t.solve()
                # dnf = sols.printN(xnames=[f"b{b}" for b in range(num_bits)], ynames=['offset'], syntax=None)
                # dnf = sols[0].printSol("offset",xnames=[f"b{b}" for b in range(num_bits)],syntax=None)
                dnf = str(
                    sols[0].expr(
                        xnames=[f"b{b}" for b in reversed(bits_used)], syntax=None
                    )
                )
                set_mapping_function = logicmin_dnf_to_sympy_cnf(dnf)
                print(set_mapping_function)

                # validate set mapping function
                valid = True
                for index_bits, offset in validation_table:
                    vars = {
                        sym.symbols(f"b{b}"): index_bits[b] for b in reversed(bits_used)
                    }
                    predicted = set_mapping_function.subs(vars)
                    predicted = int(bool(predicted))
                    if predicted != offset:
                        valid = False

                if valid:
                    # set_mapping_function = sym.logic.boolalg.to_dnf(set_mapping_function, simplify=True, force=True)
                    print(
                        color(
                            "found valid set mapping function for bit {:<2}: {}".format(
                                set_bit, set_mapping_function
                            ),
                            fg="green",
                        )
                    )
                    found_mapping_functions[set_bit] = set_mapping_function
                    break

            if found_mapping_functions.get(set_bit) is None:
                print(
                    color(
                        "no minimal set mapping function found for set bit {:<2}".format(
                            set_bit
                        ),
                        fg="red",
                    )
                )

        assert (
            str(found_mapping_functions[0])
            == "(~b13 | ~b14) & (b10 | b12 | b14 | b9) & (b10 | b12 | ~b14 | ~b9) & (b10 | b14 | ~b12 | ~b9) & (b10 | b9 | ~b12 | ~b14) & (b12 | b9 | ~b10 | ~b14) & (b14 | b9 | ~b10 | ~b12) & (b12 | ~b11 | ~b14 | ~b9) & (b11 | b12 | b14 | ~b10 | ~b9) & (b13 | b14 | ~b11 | ~b12 | ~b9) & (b11 | ~b10 | ~b12 | ~b14 | ~b9)"
        )
        assert (
            str(found_mapping_functions[1])
            == "(b10 & b13 & ~b11) | (b11 & b12 & b13 & b9) | (b11 & ~b13 & ~b9) | (b13 & ~b11 & ~b9) | (b11 & b13 & b9 & ~b10) | (b10 & b11 & ~b12 & ~b13) | (b9 & ~b10 & ~b11 & ~b13)"
        )

        found_mapping_functions_pyeda = {
            k: sympy_to_pyeda(v) for k, v in found_mapping_functions.items()
        }

        if False:
            for set_bit, f in found_mapping_functions_pyeda.items():
                minimized = pyeda_minimize(f)
                # (minimized,) = pyeda.boolalg.minimization.espresso_exprs(f.to_dnf())
                print(
                    "minimized function for set bit {:<2}: {}".format(
                        set_bit, minimized
                    )
                )

        for set_bit, f in found_mapping_functions.items():
            print("==== SET BIT {:<2}".format(set_bit))
            print_cnf_terms(f)

        for xor_expr in ["a ^ b", "~(a ^ b)"]:
            print(
                "\t{:>20}  =>  {}".format(
                    str(xor_expr),
                    str(
                        sym.logic.boolalg.to_cnf(
                            sym.parsing.sympy_parser.parse_expr(xor_expr),
                            simplify=True,
                            force=True,
                        )
                    ),
                )
            )

        if False:
            print("==== SET BIT 2 SIMPLIFIED")
            simplified1 = sym.logic.boolalg.to_cnf(
                sym.parsing.sympy_parser.parse_expr(
                    (
                        "(b10 & b13 & ~b11)"
                        "| (b11 & b12 & b13 & b9)"
                        "| (b11 ^ b13 ^ b9)"
                        "| (b11 & ~b13 & ~b9)"
                        "| (b13 & ~b11 & ~b9)"
                        "| (b10 & b11 & ~b12 & ~b13)"
                        "| (b9 & ~b10)"
                        # "| (b11 & b13 & b9 & ~b10)"
                        # "| (b9 & ~b10 & ~b11 & ~b13)"
                        # "| ((b9 | ~b10) & (b11 ^ b13))"
                    )
                ),
                simplify=True,
                force=True,
            )
            print_cnf_terms(simplified1)
            assert equal_expressions(simplified1, found_mapping_functions[1])

    # t = logicmin.TT(num_bits, 1);
    # way_boundary_addresses = range(0, known_cache_size_bytes, num_sets * known_cache_line_bytes)
    # for index, offset in zip(way_boundary_addresses, offset_bit_1):
    #     index_bits_str = np.binary_repr(index, width=num_bits)
    #     # print(index_bits_str, offset)
    #     t.add(index_bits_str, str(offset))
    #
    # sols = t.solve()
    # # print(sols.printInfo("test"))
    # # dnf = sols.printN(xnames=[f"b{b}" for b in range(num_bits)], ynames=['offset'], syntax=None)
    # # dnf = sols[0].printSol("offset",xnames=[f"b{b}" for b in range(num_bits)],syntax=None)
    # dnf = str(sols[0].expr(xnames=[f"b{b}" for b in reversed(range(num_bits))], syntax=None))
    # set_mapping_function = logicmin_dnf_to_sympy_cnf(dnf)
    # print(set_mapping_function)

    # to cnf never completes, try usign pyeda minimization
    # set_mapping_function = sym.logic.boolalg.to_cnf(set_mapping_function, simplify=True, force=True)
    # minimized = pyeda_minimize(sympy_to_pyeda(set_mapping_function))
    # print(minimized)

    for set_bit in range(num_sets_log2):
        way_boundary_addresses = range(
            0, known_cache_size_bytes, num_sets * known_cache_line_bytes
        )
        offset_bits = [
            bitarray.util.int2ba(int(o), length=num_sets_log2, endian="little")[set_bit]
            for o in offsets
        ]
        print("==== SET BIT {:<2}".format(set_bit))
        for index, offset in zip(way_boundary_addresses, offset_bits):
            full_index_bits = bitarray.util.int2ba(
                index, length=max_bits, endian="little"
            )

            # index_bits = bitarray.util.int2ba(index >> line_size_log2, length=num_bits, endian="little")
            # even xor => 0
            # uneven xor => 1

            vars = {sym.symbols(f"b{b}"): full_index_bits[b] for b in range(max_bits)}
            predicted = found_mapping_functions[set_bit].subs(vars)
            predicted = int(bool(predicted))

            # predicted = index_bits[13] + (index_bits[12] ^ index_bits[11])
            # predicted = index_bits[10] ^ index_bits[9]
            # predicted |= index_bits[8]
            # predicted = index_bits[11] ^ (index_bits[3] ^ index_bits[2])
            marks = set([3584, 7680, 11776, 19968, 24064])
            marks = set([1536])

            if False and set_bit == 0:
                predicted = (
                    index_bits[5] ^ (index_bits[4] ^ index_bits[3]) ^ index_bits[2]
                )
                # predicted = index_bits[11] ^ predicted
                if True:
                    predicted = (index_bits[4] + predicted) % 2
                    predicted = (index_bits[7] + predicted) % 2
                # ( + (index_bits[10] ^ index_bits[9])) % 2

            if set_bit == 1:
                # predicted = bool(full_index_bits[9]) ^ bool(full_index_bits[10])
                # predicted = predicted ^ bool(full_index_bits[11])
                # predicted = predicted ^ bool(full_index_bits[13])

                predicted = bool(full_index_bits[9]) ^ bool(full_index_bits[11])
                predicted = predicted ^ bool(full_index_bits[13])
                # predicted = predicted ^ bool(full_index_bits[14])
                # predicted = predicted ^ bool(full_index_bits[11])
                # predicted = predicted ^ bool(full_index_bits[13])
                # predicted |= full_index_bits[9] & full_index_bits[11])
                # predicted |= full_index_bits[10]
                # special = bool(full_index_bits[9]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13])
                if False:
                    if full_index_bits[9] & full_index_bits[10]:
                        predicted = bool(not predicted)
                        # print([full_index_bits[b] for b in [11, 12, 13, 14]])
                        # lol = bool(full_index_bits[11]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13]) ^ bool(full_index_bits[14]) ^ bool(full_index_bits[15])
                        # print(predicted, lol)
                        # predicted = predicted ^ lol
                    if (
                        full_index_bits[9]
                        & full_index_bits[10]
                        & full_index_bits[11]
                        & full_index_bits[12]
                    ):
                        predicted = bool(not predicted)

                if (
                    full_index_bits[9]
                    & full_index_bits[10]
                    & ~(full_index_bits[11] & full_index_bits[12])
                ):
                    predicted = bool(not predicted)

                # special = bool(full_index_bits[11]) ^ bool(full_index_bits[13])
                # if special:
                #     # predicted = bool(~predicted)
                #     pass
                # ^ full_index_bits[14]
                # predicted |= full_index_bits[10] ^ full_index_bits[12] ^ full_index_bits[14]

                # predicted = bool(not predicted)
                # predicted &= ~(full_index_bits[9] & full_index_bits[10])
                # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                # predicted &= special_case
                # special_case2 = ~(full_index_bits[10] & full_index_bits[11])
                # predicted |= full_index_bits[10] ^ full_index_bits[11]

            print(
                "{}\t\t{} => {:>2} {:>2} {}".format(
                    index,
                    np.binary_repr(
                        bitarray.util.ba2int(full_index_bits), width=num_bits
                    ),
                    offset,
                    str(
                        color(
                            int(predicted),
                            fg="green" if bool(predicted) == bool(offset) else "red",
                        )
                    ),
                    str(color("<==", fg="blue")) if index in marks else "",
                )
            )

    return

    offsets_df = pd.DataFrame(offsets, columns=["offset"])
    print(compute_set_probability(offsets_df, "offset"))

    offset_mapping_table = combined[["virt_addr", "set"]].copy()
    offset_mapping_table = offset_mapping_table.drop_duplicates()
    print(len(offset_mapping_table), num_sets * len(offsets_df))

    assert len(offset_mapping_table) == num_sets * len(offsets_df)
    for i in range(len(offset_mapping_table)):
        set_id = i % num_sets
        way_id = i // num_sets
        # derived_num_ways
        # print(i, set_id, way_id)
        # print(sets[:,way_id])
        set_offsets = np.argsort(sets[:, way_id])
        # print(set_offsets)
        # offsets_df
        offset_mapping_table.loc[offset_mapping_table.index[i], "set"] = int(
            set_offsets[set_id]
        )

    offset_mapping_table = offset_mapping_table.astype(int)

    # print(offset_mapping_table)
    # print("===")
    # print([int(np.binary_repr(int(s), width=2)[0]) for s in offset_mapping_table["set"]])
    # print("===")
    # print([int(np.binary_repr(int(s), width=2)[1]) for s in offset_mapping_table["set"]])

    # for way_id in range(len(offsets_df) // num_sets):
    #     way_offset = offsets_df["offset"][way_id]
    #     # print(way_id*num_sets, (way_id+1)*num_sets)
    #     rows = offset_mapping_table.index[way_id*num_sets:(way_id+1)*num_sets]
    #     offset_mapping_table.loc[rows, "set"] = way_offset

    def build_set_mapping_table(df, addr_col="virt_addr", num_sets=None, offset=None):
        set_mapping_table = df.copy()
        if offset is not None and num_sets is not None:
            set_mapping_table["set"] = (set_mapping_table["set"] + int(offset)) % int(
                num_sets
            )

        set_mapping_table = set_mapping_table[[addr_col, "set"]].astype(int)
        set_mapping_table = set_mapping_table.rename(columns={addr_col: "addr"})
        set_mapping_table = set_mapping_table.drop_duplicates()
        return set_mapping_table

    if True:
        set_mapping_table = build_set_mapping_table(offset_mapping_table)
    if False:
        # compute possible set id mappings
        # for offset in range(num_sets):
        set_mapping_table = build_set_mapping_table(combined)

    compute_set_probability(set_mapping_table)

    num_bits = 64
    print(color(f"SOLVE FOR <AND> MAPPING [bits={num_bits}]", fg="cyan"))
    sols = solve_mapping_table(set_mapping_table, use_and=True, num_bits=num_bits)
    print(color(f"SOLVE FOR <OR> MAPPING [bits={num_bits}]", fg="cyan"))
    sols = solve_mapping_table(set_mapping_table, use_and=False, num_bits=num_bits)

    # for offset in range(num_sets):
    # set_mapping_table = build_set_mapping_table(combined, num_sets=num_sets, offset=offset)

    num_bits = 64
    for degree in range(2, 4):
        print(
            color(
                f"SOLVE FOR <XOR> MAPPING [degree={degree}, bits={num_bits}]", fg="cyan"
            )
        )
        sols = solve_mapping_table_xor(
            set_mapping_table, num_bits=num_bits, degree=degree
        )
    # print(sols)

    # remove incomplete rounds
    # combined = combined[~combined["round"].isna()]

    # combined = compute_cache_lines(
    #     combined,
    #     cache_size_bytes=known_cache_size_bytes,
    #     sector_size_bytes=sector_size_bytes,
    #     cache_line_bytes=known_cache_line_bytes,
    #     )

    pass
