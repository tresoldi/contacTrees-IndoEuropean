#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Union
from copy import deepcopy
from enum import Enum

import pandas as pd
from newick import Node, parse_node

from contacTreesIE.newick_util import (
    get_sibling,
    translate_node_names,
    get_age,
    get_node_by_name,
)
from contacTreesIE.preprocessing.language_lists import *
from contacTreesIE.preprocessing import xml_snippets
from contacTreesIE.preprocessing.xml_snippets import Samplers
from contacTreesIE.preprocessing.starting_trees import CHANG_MEDIUM_TREE
from contacTreesIE.preprocessing.starting_trees import CHANG_MEDIUM_TRANSLATE
from contacTreesIE.preprocessing.starting_trees import (
    GERARDI_TUPI_TREE,
    GERARDI_TRANSLATE,
)

XML_TEMPLATE_PATH = "resources/ie_template.xml"
LOANS_CSV_PATH = "loans.csv"
ZOMBIE_LATIN = "Latin_preserved"
MEDIEVAL_LATIN = "Latin_M"


tg_langs = [
    "Ache",
    "Amondawa",
    "Anambe",
    "Apiaka",
    "Arawete",
    "Asurini_Tocantins",
    "Asurini_Xingu",
    "Ava_Canoeiro",
    "Aweti",
    "Chiriguano",
    "Guajajara",
    "Guaja",
    "Guarani",
    "Guarayo",
    "Kaapor",
    "Kaiowa",
    "Kamajura",
    "Kayabi",
    "Kokama",
    "Mawe",
    "Mbya",
    "Nheengatu",
    "Old_Guarani",
    "Omagua",
    "Parakana",
    "Parintintin",
    "Siriono",
    "Surui_Aikewara",
    "Tapiete",
    "Tapirape",
    "Teko",
    "Tembe",
    "Tenharim",
    "Tupinamba",
    "Urueuwauwau",
    "Warazu",
    "Wayampi",
    "Xeta",
    "Yuki",
    "Zoe",
]


# DATASET = CHANG_MEDIUM
# DATASET = TINY_SET
# INCLUDED_CLADES = CELTIC + GERMANIC + ROMANCE
INCLUDED_LANGUAGES = tg_langs
RENAME = RENAME_GERARDI
translate = GERARDI_TRANSLATE
STARTING_TREE = GERARDI_TUPI_TREE

SA_PRIOR_SIGMA = 100.0

TIP_DATES = defaultdict(float)
TIP_DATES.update(
    {
        "Old_Guarani": 325.0,
        "Tupinamba": 400.0,
        "Guarayo": 75.0,
        "Xeta": 50.0,
    }
)

SAMPLED_ANCESTORS = [
    # "Tupinambá",
]


class CalibrationModes(Enum):
    BOUCKAERT = 1
    CHANG = 2


CALIBRATION_MODE = CalibrationModes.BOUCKAERT


class Clade(object):

    clade_template = """\
        <distribution id="{clade}.prior" spec="MRCAPrior" tree="@acg" monophyletic="true">
          <taxonset id="{clade}" spec="TaxonSet">
{taxa}
          </taxonset>
{age_prior}        </distribution>\n\n"""

    taxon_template = '            <taxon idref="{taxon}" />'

    def __init__(self, name: str, members: list, age_prior: Union[str, dict] = ""):
        self.name = name
        self.members = [
            m for m in members if isinstance(m, Clade) or (m in INCLUDED_LANGUAGES)
        ]
        # self.members = members
        if isinstance(age_prior, dict):
            self.age_prior = age_prior[CALIBRATION_MODE]
        else:
            self.age_prior = age_prior

    def get_languages(self) -> List[str]:
        languages = []
        for m in self.members:
            if isinstance(m, Clade):
                languages += m.get_languages()
            else:
                assert isinstance(m, str)
                languages.append(m)
        return languages

    def to_xml(self):
        taxa_str = "\n".join(
            self.taxon_template.format(taxon=l) for l in self.get_languages()
        )

        age_prior = ""
        if self.age_prior:
            age_prior = (" " * 10) + self.age_prior + "\n"

        return self.clade_template.format(
            clade=self.name, taxa=taxa_str, age_prior=age_prior
        )


AwetiTG = Clade(
    name="AwetiTG",
    members=[l for l in tg_langs if l != "Mawe"],
)

TG = Clade(
    name="TG",
    members=[l for l in tg_langs if l not in ["Mawe", "Aweti"]],
)

GTG = Clade(
    name="GTG",
    members=["Guaja", "Guajajara", "Tembe"],
)

Guaranian = Clade(
    name="Guaranian",
    members=[
        "Ache",
        "Chiriguano",
        "Guarani",
        "Guarayo",
        "Kaiowa",
        "Mbya",
        "Old_Guarani",
        "Siriono",
        "Tapiete",
        "Warazu",
        "Xeta",
        "Yuki",
    ],
)

Kawahiva = Clade(
    name="Kawahiva",
    members=["Amondawa", "Apiaka", "Kayabi", "Parintintin", "Tenharim", "Urueuwauwau"],
)

OK = Clade(
    name="OK",
    members=["Omagua", "Kokama"],
)

Tupi = Clade(
    name="Tupi",
    members=["Omagua", "Kokama", "Nheengatu", "Tupinamba"],
)

ZWT = Clade(
    name="ZWT",
    members=["Teko", "Wayampi", "Zoe"],
)


CLADES = [
    AwetiTG,
    TG,
    GTG,
    Guaranian,
    Kawahiva,
    OK,
    Tupi,
    ZWT,
]


def read_dataset(csv_path):
    """Read the cognate data-set from a TSV file and run some basic parsing."""
    tg = pd.read_csv(csv_path, sep="\t", dtype=str)
    # try:
    #    tg["CONCEPT"] = tg["COGID"].map(lambda s: s.split("-")[0].strip())
    # except AttributeError as e:
    #    for cogid in tg.COGID:
    #        if not isinstance(cogid, str) or ("-" not in cogid):
    #            print("Invalid cogid:", cogid)
    #    raise e

    return tg


def drop_noncoded(ielex):
    """Drop rows which do not have a cognate ID."""
    return ielex.loc[~pd.isna(ielex.COGID)]


def parse_loanwords(ielex: pd.DataFrame):
    loans = pd.DataFrame(
        data=0, dtype=int, columns=ielex.DOCULECT.unique(), index=ielex.CONCEPT.unique()
    )

    for concept, concepts_grouped in ielex.groupby("CONCEPT"):
        concepts_grouped.set_index("DOCULECT")
        for i_cc, (_, cc_grouped) in enumerate(concepts_grouped.groupby("COGID")):
            for _, row in cc_grouped.iterrows():
                if row.status in ("LOAN",):
                    loans.loc[concept, row.language] = 1
    return loans


def parse_data_matrix(ielex: pd.DataFrame, exclude_loans=False):
    """Parse the pandas dataframe, containing the IELEX data into nested dictionary, which
    is easier to use for compiling the BEAST XML.

    Args:
        ielex (pd.DataFrame): the Pandas DataFrame containing the raw IELex dataset.
        exclude_loans (boolean): Whether to exclude loan words from the analysis.

    Returns:
        dict: The parsed data set which is nested dictionary:
                {language (str) ->  {concept (str) -> data (string of 0/1/?)}}
    """
    languages = ielex.DOCULECT.unique()

    # ´data´ is a nested dict:
    # {language: {concept: "absence/presence of each cognate for this concept as a binary string"}}
    data = {lang: {} for lang in languages}

    for concept, concepts_grouped in ielex.groupby("CONCEPT"):
        concepts_grouped.set_index("DOCULECT")
        n_cc = len(concepts_grouped.COGID.unique())
        concept_data = defaultdict(list)

        # Collect all cognates for the current concept
        for i_cc, (_, cc_grouped) in enumerate(concepts_grouped.groupby("COGID")):
            for _, row in cc_grouped.iterrows():
                # if row.status in ("EXCLUDE", "LOAN,EXCLUDE", "WRONG"):
                #    continue
                # elif exclude_loans and row.status == "LOAN":
                #    continue
                # elif not pd.isna(row.status):
                #    raise ValueError(
                #        f'Unknown status "{row.status}" in {row.language}:{row.cc_alias}'
                #    )
                concept_data[row.DOCULECT].append(i_cc)

        # Assemble cognate absence/presence for the current concept and each language in a binary string
        for lang in languages:
            if lang in concept_data:
                data[lang][concept] = "".join(
                    ["1" if i_cc in concept_data[lang] else "0" for i_cc in range(n_cc)]
                )
                assert "1" in data[lang][concept]
            else:
                data[lang][concept] = "".join(["?" for _ in range(n_cc)])

    return data


def encode_binary_array(a):
    """Compact bit-string encoding for binary arrays."""
    return "".join(["%i" % x for x in a])


def compile_ielex_xml(
    data_raw: dict,
    ascertainment_correction=True,
    min_coverage=0.0,
    fixed_topolgy=False,
    fixed_node_heights=False,
    use_contactrees=True,
    expected_conversions=0.25,
    exclude_loans=False,
    add_zombie_latin=False,
    add_medieval_latin=False,
    use_covarion=True,
    sample_acg_prior_params=False,
    clock_stdev_prior=0.04,
    fix_clock_stdev=True,
    sampler=Samplers.MCMC,
    chain_length=20000000,
):
    """Compile the IELEX data together with hardcoded settings into a BEAST XML.

    Args:
        data_raw (pd.DataFrame): The pandas data-frame containing the raw IELex data.
        ascertainment_correction (bool): whether to apply ascertainment correction.
        min_coverage (float): the minimum fraction of cognates a language need to be included.
        use_contactrees (bool): whether use contacTrees for the reconstruction.
        fixed_topolgy (bool): whether to fix the tree topology.
        fixed_node_heights (bool): whether to fix the height of internal nodes in the tree.
        expected_conversions (float): The expected number of conversions in the prior.
        exclude_loans (bool): whether to exclude loans from the data or not.
        add_zombie_latin (bool): whether to add a preserved copy of the Latin taxon as
            a contemporary language to allow for recent borrowing of Latin words.

    Returns:
        str: The compiled BEAST2 XML file.
    """
    clades = deepcopy(CLADES)
    data = parse_data_matrix(data_raw, exclude_loans=exclude_loans)

    if add_medieval_latin:
        INCLUDED_LANGUAGES.append(MEDIEVAL_LATIN)

        for clade in clades:
            if "Latin" in clade.members:
                clade.members.append(MEDIEVAL_LATIN)

        clades.append(
            Clade(
                name="Latin_descendants",
                members=["Latin", MEDIEVAL_LATIN],
                age_prior='<distr id="Latin_SA.tmrca" spec="Normal" offset="2050" sigma="20"/>',
            )
        )

    if add_zombie_latin:
        INCLUDED_LANGUAGES.append(ZOMBIE_LATIN)

        # Use the most recent form of latin as data for zombie latin
        if add_medieval_latin:
            data[ZOMBIE_LATIN] = data[MEDIEVAL_LATIN]
        else:
            data[ZOMBIE_LATIN] = data["Latin"]

        for clade in clades:
            if "Latin" in clade.members:
                clade.members.append(ZOMBIE_LATIN)

    if add_medieval_latin and add_zombie_latin:
        clades.append(
            Clade(
                name="Latin_M_descendants",
                members=[MEDIEVAL_LATIN, ZOMBIE_LATIN],
                age_prior='<distr id="Latin_M_SA.tmrca" spec="Normal" offset="1000" sigma="10"/>',
            )
        )

    language_alignments = defaultdict(str)
    concept_ranges = {}
    concept_n_sites = {}
    coverage = defaultdict(int)

    n_concepts = 0
    words_per_language = defaultdict(int)
    for lang, data_l in data.items():
        i = 0
        for concept, data_l_c in data_l.items():
            is_nan = data_l_c.startswith("?")
            if not is_nan:
                coverage[lang] += 1
                words_per_language[lang] += sum(map(int, data_l_c))

            if ascertainment_correction:
                if is_nan:
                    data_l_c = "?" + data_l_c
                else:
                    data_l_c = "0" + data_l_c

            language_alignments[lang] += data_l_c

            if concept not in concept_ranges:
                i_next = i + len(data_l_c)
                concept_ranges[concept] = (i, i_next - 1)
                concept_n_sites[concept] = len(data_l_c)
                i = i_next
                n_concepts += 1

    # Filter languages with insufficient data
    for lang in list(language_alignments.keys()):
        if lang not in INCLUDED_LANGUAGES:
            language_alignments.pop(lang)

    # Fill in alignments for each language
    alignments = ""
    for lang, alignment in language_alignments.items():
        if coverage[lang] < min_coverage * n_concepts:
            print("Coverage too low:", lang, coverage[lang])
            continue

        alignments += xml_snippets.ALIGNMENT.format(tax_name=lang, data=alignment)

    # Fill in filtered alignments for each block
    filtered_alignments = ""
    for concept, (start, end) in concept_ranges.items():
        filtered_alignments += xml_snippets.FILTERED_ALIGNMENT.format(
            concept=concept,
            start=start + 1,
            end=end + 1,
            excludeto=int(ascertainment_correction),
        )

    # Compile list of tip-dates
    format_date = lambda l: f"        {l} = {TIP_DATES[l]}"
    dates = ",\n".join(map(format_date, language_alignments.keys()))

    # Compile MCRA priors
    mrca_priors = "".join(clade.to_xml() for clade in clades)

    # Compile likelihoods
    # with slow, medium or fast clock depending on the number of sites per concept
    likelihoods = ""
    for concept, n_sites in concept_n_sites.items():
        if n_sites <= 5:
            site_category = "slow"
        elif n_sites <= 9:
            site_category = "medium"
        else:
            site_category = "fast"

        if use_contactrees:
            likelihoods += xml_snippets.CONTACTREES_LIKELIHOOD.format(
                concept=concept, site_cat=site_category
            )
        else:
            likelihoods += xml_snippets.BASICTREES_LIKELIHOOD.format(
                concept=concept, site_cat=site_category
            )
            expected_conversions = 0.0

    frozen_taxa = xml_snippets.FROZEN_TAXA if add_zombie_latin else ""

    if use_covarion:
        substitution_model = xml_snippets.COVARION_MODEL
        substitution_model_prior = xml_snippets.COVARION_PRIORS
        data_type = xml_snippets.COVARION_DATA_TYPE
    else:
        substitution_model = xml_snippets.CTMC_MODEL
        substitution_model_prior = xml_snippets.CTMC_PRIORS
        data_type = xml_snippets.CTMC_DATA_TYPE

    # Compile operators
    operators = "\n"
    if fixed_node_heights:
        assert fixed_topolgy
    if not fixed_topolgy:
        operators += xml_snippets.TOPOLOGY_OPERATORS
    if not fixed_node_heights:
        operators += xml_snippets.NODE_HEIGHTS_OPERATORS
    if use_contactrees:
        operators += xml_snippets.CONTACT_OPERATORS

    # Prepare the starting tree
    starting_tree = fix_tree(STARTING_TREE, add_zombie_latin, add_medieval_latin)

    # Add word-tree loggers if required
    if use_contactrees:
        # word_tree_loggers = xml_snippets.WORD_TREE_LOGGERS
        word_tree_loggers = ""
    else:
        word_tree_loggers = ""

    # Prepare the `run` tag
    run_tag = xml_snippets.SAMPLER_TAG[sampler].format(chain_length=chain_length)

    # Load the BEAST2 XML template
    with open(XML_TEMPLATE_PATH, "r") as xml_template_file:
        xml_template = xml_template_file.read()

    if fix_clock_stdev:
        clock_stdev_operator = ""
    else:
        clock_stdev_operator = xml_snippets.CLOCK_STDEV_OPERATOR

    # Put everything together...
    xml_filled = xml_template.format(
        concepts=",".join(concept_ranges.keys()),
        languages=",".join(language_alignments.keys()),
        alignments=alignments,
        filtered_alignments=filtered_alignments,
        dates=dates,
        mrca_priors=mrca_priors,
        likelihood=likelihoods,
        clock_rate_prior=xml_snippets.CLOCK_RATE_PRIOR_FLAT,
        clock_stdev_prior=clock_stdev_prior,
        subst_model=substitution_model,
        subst_model_prior=substitution_model_prior,
        data_type=data_type,
        operators=operators,
        clock_stdev_operator=clock_stdev_operator,
        starting_tree=starting_tree,
        expected_conversions=str(expected_conversions),
        frozen_taxa=frozen_taxa,
        word_tree_loggers=word_tree_loggers,
        run_tag=run_tag,
    )

    return xml_filled


def fix_tree(old_newick: str, add_zombie_latin: bool, add_medieval_latin: bool) -> str:
    old_newick_no_attr = drop_attributes(old_newick)

    tree = parse_node(old_newick_no_attr.strip(" ;"))
    translate_node_names(tree, translate)
    translate_node_names(tree, RENAME_GERARDI)

    for name in INCLUDED_LANGUAGES:
        if name in [ZOMBIE_LATIN, MEDIEVAL_LATIN]:
            continue
        assert name in tree.get_leaf_names(), name

    tree.prune_by_names(INCLUDED_LANGUAGES, inverse=True)
    tree.remove_redundant_nodes()
    tree.length = 0.0
    tree = parse_node(tree.newick)

    for node in tree.get_leaves():
        err = get_age(node) - TIP_DATES[node.name]

        node.length += err
        if node.length < 0:
            node.ancestor.length += node.length - 1.0
            sibling = get_sibling(node)
            if sibling is not None:
                sibling.length -= node.length - 1.0
            node.length = 1.0

    if add_medieval_latin:
        latin: Node = get_node_by_name(tree, "Latin")
        old_parent: Node = latin.ancestor
        medieval_latin: Node = Node.create(
            name=MEDIEVAL_LATIN,
            length="%.8f" % (get_age(latin) + 1.0 - TIP_DATES[MEDIEVAL_LATIN]),
        )
        new_parent: Node = Node.create(
            name="",
            length="%.8f" % (latin.length - 1.0),
            descendants=[latin, medieval_latin],
        )
        latin.length = 1.0
        old_parent.descendants.remove(latin)
        old_parent.add_descendant(new_parent)

    if add_zombie_latin:
        if add_medieval_latin:
            parent = get_node_by_name(tree, MEDIEVAL_LATIN)
        else:
            parent = get_node_by_name(tree, "Latin")

        old_parent: Node = parent.ancestor
        zombie_latin: Node = Node.create(
            name=ZOMBIE_LATIN, length="%.8f" % (get_age(parent) + 1.0)
        )
        new_parent: Node = Node.create(
            name="",
            length="%.8f" % (parent.length - 1.0),
            descendants=[parent, zombie_latin],
        )
        parent.length = 1.0
        old_parent.descendants.remove(parent)
        old_parent.add_descendant(new_parent)

    return tree.newick


def drop_attributes(newick):
    while "[" in newick:
        before_attr, _, rest = newick.partition("[")
        attr_str, _, after_attr = rest.partition("]")
        newick = before_attr + after_attr
    return newick


def filter_languages(df):
    include = df.DOCULECT.isin(INCLUDED_LANGUAGES)
    return df[include]


if __name__ == "__main__":
    DATA_PATH = Path("tupi/tuled_norm.csv")
    tg_df = read_dataset(DATA_PATH)
    tg_df = drop_noncoded(tg_df)

    path_base, _, path_ext = str(DATA_PATH).rpartition(".")
    subset_path = path_base + "-subset.tsv"
    tg_df = filter_languages(tg_df)
    tg_df.to_csv(subset_path, sep="\t", index=False)

    N_RUNS = 5

    RUN_CONFIGURATIONS = {
        "tupi_BT_fixTopo/covarion": {
            "sampler": Samplers.MCMC,
            "chain_length": 20000000,
            "use_contactrees": False,
            "fixed_topolgy": True,
            "fixed_node_heights": False,
            "exclude_loans": False,
            "add_zombie_latin": False,
            "use_covarion": True,
        },
        #    "BT_fixTopo/covarion_noLoans": {
        #        "sampler": Samplers.MCMC,
        #        "chain_length": 20000000,
        #        "use_contactrees": False,
        #        "fixed_topolgy": True,
        #        "fixed_node_heights": False,
        #        "exclude_loans": True,
        #        "add_zombie_latin": True,
        #        "use_covarion": True,
        #    },
        #    "BT_full": {
        #        "sampler": Samplers.MCMC,
        #        "chain_length": 20000000,
        #        "use_contactrees": False,
        #        "fixed_topolgy": False,
        #        "fixed_node_heights": False,
        #        "exclude_loans": False,
        #        "add_zombie_latin": True,
        #        "use_covarion": True,
        #    },
        #    "CT_fixTopo/covarion": {
        #        "sampler": Samplers.MC3,
        #        "chain_length": 20000000,
        #        "use_contactrees": True,
        #        "fixed_topolgy": True,
        #        "fixed_node_heights": False,
        #        "exclude_loans": False,
        #        "add_zombie_latin": True,
        #        "use_covarion": True,
        #    },
        #    "CT_full": {
        #        "sampler": Samplers.MC3,
        #        "chain_length": 25000000,
        #        "use_contactrees": True,
        #        "fixed_topolgy": False,
        #        "fixed_node_heights": False,
        #        "exclude_loans": False,
        #        "add_zombie_latin": True,
        #        "use_covarion": True,
        #    },
    }

    for run_name, kwargs in RUN_CONFIGURATIONS.items():

        # Create folder structure
        run_directory = Path(f"runs/fix_clock_stdev/{run_name}")
        os.makedirs(run_directory, exist_ok=True)

        # Compile the BEAST-XML from the data and the run-configuration
        ie_xml_str = compile_ielex_xml(tg_df, **kwargs)

        # Write the XML to ´N_RUNS different files´
        fname_base = run_name.replace("/", "_")
        for i in range(1, N_RUNS + 1):
            fname = f"{fname_base}_{i}.xml"
            with open(run_directory / fname, "w") as ie_xml_file:
                ie_xml_file.write(ie_xml_str)

        # Create a shell script to start all runs in parallel
        with open(run_directory / "start_runs.sh", "w") as ie_xml_file:
            lines = []
            for i in range(1, N_RUNS + 1):
                threads = xml_snippets.SAMPLER_THREADS[kwargs["sampler"]]
                lines.append(
                    f"beast -threads {threads} -overwrite {fname_base}_{i}.xml > {fname_base}_{i}.screenlog &"
                )
            ie_xml_file.write("\n".join(lines))

    # loans = parse_loanwords(tg_df)
    # loans.to_csv(LOANS_CSV_PATH)
