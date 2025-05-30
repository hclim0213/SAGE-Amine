"""
Copyright (c) 2024 Hocheol Lim.
"""
# This code was from MolGpKa (https://github.com/Xundrug/MolGpKa)

import numpy as np
from rdkit import Chem

import os
import pandas as pd
from io import StringIO

smarts_data = """Substructure	    SMARTS	  Index   	Acid_or_base
1	[SX4:0](=[O:1])(=[O:2])([O:3]-[C,c,N,n:4])-[OX2:5]-[H:6]	6	A
2	[SX4:0](=[O:1])(=[O:2])(-[C,c,N,n:3])-[OX2:4]-[H:5]	5	A
3	[SX3:0](=[O:1])-[O:2]-[H:3]	3	A
4	[c,n,o:0]-[C:1](=[O:2])-[O:3]-[H:4]	4	A
5	[C:0](=[O:1])-[O:2]-[H:3]	3	A
6	[C,c,N,n:0](=[O,S:1])-[SX2,OX2:2]-[H:3]	3	A
7	[c,n:0]-[SX2:1]-[H:2]	2	A
8	[C,N:0]-[SX2:1]-[H:2]	2	A
9	[PX4:0](=[O:1])(-[OX2:2]-[H:3])(-[O+0:4])-[OX2:5]-[H:6]	3,6	A
10	[PX4:0](=[O:1])(-[OX2:2]-[H:3])(-[C,c,N,n:4])-[OX2:5]-[H:6]	3,6	A
11	[PX4:0](=[O:1])(-[O:2]-[H:3])(-[H:4])-[C:5]	3	A
12	[c,n,o:0]-[O:1]-[H:2]	2	A
13	[O:0]([$(C=O),$(C[Cl]),$(CF),$(C[Br]),$(CC#N):1])-[O:2]-[H:3]	3	A
14	[C:0]-[O:1]-[O:2]-[H:3]	3	A
15	[O:0]=[C;R:1]-[C;R:2]=[C;R:3]-[O:4]-[H:5]	5	A
16	[C:0]=[C:1]-[O:2]-[H:3]	3	A
17	[C:0]-[O:1]-[H:2]	2	A
18	[C:0](=[O:1])-[N:2]-[O:3]-[H:4]	4	A
19	[O,S:0]=[C;R:1]([$([#8]),$([#7]),$([#16]),$([#6][Cl]),$([#6]F),$([#6][Br]):2])-[N;R:3]([C;R:4]=[O,S:5])-[H:6]	6	A
20	[O,S:0]=[C;R:1]-[N;R:2]([C;R:3]=[O,S:4])-[H:5]	5	A
21	[F,Cl,Br,S,s,P,p:0][#6:1][CX3:2](=[O,S:3])-[NX3+0:4]([CX3:5]=[O,S:6])-[H:7]	7	A
22	[O,S:0]=[CX3:1]-[NX3+0:2]([CX3:3]=[O,S:4])-[H:5]	5	A
23	[C:0](=[O:1])-[N:2](-[Br,Cl,I,F,S,O,N,P:3])-[H:4]	4	A
24	[C:0](=[O:1])-[N:2]-[H:3]	3	A
25	[SX4:0](=[O:1])(=[O:2])-[NX3+0:3]-[H:4]	4	A
26	[PX4:0](=[O:1])(-[C,c,N,n,F,Cl,Br,I:2])(-[C,c,N,n,F,Cl,Br,I:3])-[OX2:4]-[H:5]	5	A
27	[PX4:0](=[O:1])(-[OX2:2]-[C,c,N,n,F,Cl,Br,I:3])(-[O+0:4]-[C,c,N,n,F,Cl,Br,I:5])-[OX2:6]-[H:7]	7	A
28	[PX4:0](=[O:1])(-[OX2:2]-[C,c,N,n,F,Cl,Br,I:3])(-[C,c,N,n,F,Cl,Br,I:4])-[OX2:5]-[H:6]	6	A
29	[n:0]-[H:1]	1	A
30	[PX4:0](=[O:1])(=[O:2])(-[C,c,N,n:3])-[OX2:4]-[H:5]	5	A
31	[C:0](-[C,c:1])=[NX2:2]-[O:3]-[H:4]	4	A
32	[CX4:0]-[NX3:1](-[H:2])-[CX3:3](=[NX2:4])-[SH:5]	2	A
33	[CX3:0](=[OX1:1])-[CX4:2]-[NX3:3](-[H:4])-[cX3:5]	4	A
34	[CX3:0](=[OX1:1])-[CX4:2]-[NX3H1:3](-[H:4])-[CX3:5]=[NX2H0:6]	4	A
35	[NX3:0](-[H:1])-[C:2]=[C:3]-[C:4](=[O:5])-[C:6]-[O:7]	1	A
36	[c:0]-[NX3:1](-[H:2])-[C,n:3]	2	A
37	[NX3:0]-[OX2:1]-[H:2]	2	A
38	[SX4:0](=[O:1])(=[O:2])-[NX3:3]-[H:4]	4	A
39	[NX2:0]-[NX2:1]=[C:2]-[NX3:3]-[H:4]	4	A
40	[C:0]-[NX3:1](-[H:2])-[C:3](=[SX1:4])	2	A
41	[C:0](-[C:1])(-[C:2])=[NX2:3]-[H:4]	4	A
42	[C:0]=[NX2:1]-[NX3:2](-[H:3])-[c:4]	3	A
43	[C:0]-[C:1](-[C,O:2])=[NX2:3]-[H:4]	4	A
44	[cX3;r6:0]-[NX3:1]-[H:2]	2	A
45	[C:0](=[O:1])-[C:2](-[H:3])(-[C:4])-[C:5](=[O:6])	3	A
46	[PX4:0](=[O:1])(=[O:2])-[O:3]-[H:4]	4	A
47	[C:0](-[H:1])(-[C:2]=[O:3])-[C:4]=[O:5]	1	A
48	[C:0](=[S,O:1])-[NX3:2]-[H:3]	3	A
49	[S:0]-[S:1](=[O:2])(=[O:3])-[O:4]-[H:5]	5	A
50	[C:0]=[N:1]-[O:2]-[H:3]	3	A
51	[C:0](-[N:1])(=[N:2])-[N:3](-[C:4])-[H:5]	5	A
52	[NX2H0;r6:0]=[C;r6:1]-[C;r6:2](-[H:3])-[C:4]#[N:5]	3	A
53	[C:0]=[NX2:1]-[H:2]	2	A
54	[c,C:0]-[C:1](=[OX1:2])-[C:3]=[C:4]-[NX3:5]-[H:6]	6	A
55	[N:0]-[C:1](-[N:2])=[NX2:3]-[H:4]	3	B
56	[C:0](-[N:1])=[NX2+0:2]-[H:3]	2	B
57	[c:0]-[NX3+0:1]([H:2])[H:3]	1	B
58	[c:0]-[NX3+0:1]([H:2])[!H:3]	1	B
59	[c:0]-[NX3+1:1]([!H:2])([!H:3])-[H:4]	1	B
60	[n+1&H1:0]	0	B
61	[C:0]-[NX4+1:1]-[H:2]	1	B
62	[C,c:0]-[O:1]-[NH2:2]	2	B
63	[CX4:0]-[NX3:1]([CX4,c:2])([CX4,c:3])	1	B
64	[C:0](-[C,c:1])=[NX2:2]-[O:3]-[H:4]	2	B
65	[C:0]-[nX3;H0:1]([C,c:2])([C,c:3])	1	B
66	[c:0]:[nX2:1]:[c:2]	1	B
67	[c:0]:[nX2:1]:[nX3:2]:[c:3]=[OX1:4]	1	B
68	[CX4:0]-[NX3:1](-[H:2])-[CX4:3]	1	B
69	[C:0](=[O:1])-[N:2](-[H:3])-[NX3:4](-[H:5])(-[H:6])	4	B
70	[c:0]:[nX2:1]:[n:2]:[n:3]:[n:4]-[CX4:5]	1	B
71	[CX4:0](-[CX4:1])(-[CX4:2])-[NH2:3]	3	B
72	[CX3:0]=[NX2:1]-[CX4,SX4:2]	1	B
73	[c:0]-[NH2:1]	1	B
74	[!OX1:0][C,c:1]-[NX3:2](-[c:3])-[H:4]	2	B
75	[C,c:0]-[NX2:1]=[c:2]:[nX3:3]-[H:4]	1	B
76	[nX2:0](:[c:1]):[nX3:2]-[c:3]	0	B
77	[c:0]=[NX2:1]-[H:2]	1	B
78	[c:0]:[nX2:1]:[nX2:2]:[nX3:3](-[H:4]):[nX2:5]	1	B
79	[OX1:0]=[CX3:1]-[C:2]-[NH2:3]	3	B
80	[C:0]=[N:1]-[N:2]=[C:3](-[NH2:4])-[S,O,P:5]	2	B
81	[nH1:0]:[c:1]:[nX2H0:2]:[nX2H0:3]:[cX3:4]:[cX3:5]	2,3	B
82	[cX3:0]:[nX2H0:1]:[nX2H0:2]:[cX3:3]:[sX2H0:4]	1,2	B
83	[cX3:0]:[cX3:1]:[nX2H0:2]:[nX2H0:3]:[cX3:4]:[cX3:5]:[cX3:6]:[nX3:7]:[cX3:8]	2,3	B
84	[NX3H0:0]-[CX3:1]=[NX2H0:2]-[NX2H0:3]=[CX3:4]	2	B
85	[CX3:0]=[NX2H0:1]-[NX3H1:2]-[CX4:3]	1,2	B
86	[CX3:0](=[OX1:1])-[CX4:2]-[NX3:3](-[H:4])-[cX3:5]	3	B
87	[CX3:0](=[OX1:1])-[CX4:2]-[NX3H1:3](-[H:4])-[CX3:5]=[NX2H0:6]	6	B
88	[C:0]-[NH:1]-[NH:2]-[C:3]=[O:4]	1	B
89	[C:0]=[NX2:1]-[OX2:2]	1	B
90	[PX4:0](=[OX1:1])(-[NH2:2])(-[NH1:3])	2,3	B
91	[c:0]:[nX2H0:1]:[nX2H0:2]:[cX3:3]:[nX3:4]-[NX3:5]	1,2	B
92	[c:0]:[nX2:1]:[nX3:2]:[cX3:3]:[cX3:4]	1	B
93	[C:0]=[NX2:1]-[C:2]	1	B
94	[PX4:0](-[NH1:1])=[OX1:2]	1	B
95	[CX4:0]-[CX4:1]-[NH2:2]	2	B
96	[c:0]=[NX2:1]	1	B
97	[PX4:0]-[NX3H0:1]	1	B
98	[C:0]-[NX3:1]-[C:2](=[SX1:3])-[NX3:4]-[C:5]	3	B
99	[OX2:0]-[C:1](=[OX1:2])-[NX3:3]-[NX3:4]	4	B
100	[c:0]-[NX2:1]=[C:2]	1	B
101	[c,S:0]-[C:1]-[NH2:2]	2	B
102	[NX3:0]-[OX2:1]-[H:2]	0	B
103	[c:0]:[n:1]:[n:2]:[c:3]:[nX3:4]-[c:5]	1,2	B
104	[C:0]-[C:1]-[NH2:2]	2	B
105	[S:0]-[NH1:1]-[C:2]	1	B
106	[c:0]-[NX3:1](-[H:2])-[c:4]	1	B
107	[C:0]=[NX2:1]-[NX2:2]=[C:3]-[NH2:4]	2	B
108	[C:0]-[C:1]-[C:2]=[NX2H0:3]-[NX3H0:4]	3,4	B
109	[c:0]-[NX3H1:1]-[C:2]-[C:3]=[C:4]-[c:5]	1	B
110	[NX3:0]-[C:1](=[S:2])-[NX3:3]	2	B
111	[c:0]:[nH0:1]:[nH0:2]:[c:3]:[c:4]:[c:5]	1,2	B
112	[c:0]:[c:1]-[nX3H1:2]-[C:3]-[C:4]=[C:5]	2	B
113	[C:0]=[C:1]-[NX3H0:2](-[C:3]-[C:4])(-[C:5]-[C:7])	2	B
114	[nX3:0]:[c:1]:[nX2:2]:[nX2:3]:[c:4]	2,3	B
115	[n:0]-[NX3H2:1]	1	B
116	[nX3H1:0]:[c:1]:[c:2]:[c:3](=[O:4]):[c:5]:[c:6]	4	B
117	[C:0]-[NX3H1:1]-[NX3H2:3]	1,2	B
118	[OX2:0]-[C:1]-[NX3H2:2]	2	B
119	[c:0]:[nX2H0:1]:[nX2H0:2]:[nX3H0:3]:[c:4]	1	B
120	[SX4:0](=[O:1])(=[O:2])-[NX3H2:3]	3	B
121	[NX3H0:0](-[C;!CSX1:1])(-[C;!CSX1:2])-[C;!CSX1:3]	0	B
122	[c:0]:[c:1]:[c:2]:[c:3](-[OH:4]):[nX3H1:5]	5	B
123	[C:0]-[C:1]-[NX3H0:2]-[OX2:3]-[C:4]	2	B
124	[c:0]:[c:1]-[NX3H1:2]-[C:3]-[C:4]-[C:5]-[c:6]	2	B
125	[C:0](=[NX2H1:1])(-[C:2])-[C:3]	1	B
126	[C,c:0]-[NX2H0:1]=[NX2H0:2]-[NX3:3]	3	B
127	[c:0]:[c:1]:[c:2]:[c:3]:[nX3:4]-[cX3:5]	4	B
128	[nX2H0:0]:[nX2H0:1]:[nX2H0:2]:[nX3H0:3]:[c:4]:[c:5]	0	B
129	[C:0]=[NX2:1]-[NH1:2]-[c:3]	1	B
130	[C:0]-[C:1](-[C,O:2])=[NH1:3]	3	B
131	[C:0]-[NX3H0:1](-[C:2])-[NX3H1:3]-[C:4]	1	B
132	[nX2:0]:[nX2:1]:[cX3:2](-[C:3]):[nX2:4]:[nX3:5]-[C:6]	1	B
133	[c:0]-[C:1]=[NX2H0:2]-[NX2H0:3]=[C:4]-[SH:5]	3	B
134	[c:0]-[c:1]:[nX2H0:2]:[nX2H0:3]:[nX3:4](-[C:5]):[nX2:6]	2	B
135	[c:0](=[SX1:1]):[nX3H1:2]:[c:3]	1,2	B
136	[C:0]=[NX2H0:1]-[NX3H0:2](-[C:3])-[C:4]	1,2	B
137	[c:0]:[nX2H0:1]:[sX2H0:2]:[c:3]:[c:4]	1	B
138	[C:0]=[NX2H0:1]-[NH2:2]	1,2	B
139	[nX2H0:0]:[nX2H0:1]:[c:2]:[nH1:3]:[c:4]:[c:5]:[c:6]	1	B
140	[NX2H0;r7:0]-[NX2H0;r7:1]=[C;r7:2]-[NH1;r7:3]-[c;r7:4]:[c;r7:5]-[C;r7:6]	1	B
141	[NX3:0](-[C:1])(-[C:2])-[C:3]=[S:4]	0	B
142	[c:0]:[nX2H0:1]:[nX2H0:2]:[nX2H0:3]:[nX3H0:4]	1	B
143	[c;r5:0]:[c;r5:1]:[c;r5:2]:[c;r5:3]:[nH1;r5:4]	4	B
"""

smarts_file = "smarts_pattern.tsv"

def split_acid_base_pattern():
    df_smarts = pd.read_csv(StringIO(smarts_data), sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base

def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m)==1]))
    double_matches = [m for m in matches if len(m)==2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches

def match_acid(df_smarts_acid, mol):
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_acid.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def match_base(df_smarts_base, mol):
    matches = []
    for idx, name, smarts, indexs, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        index_split = indexs.split(",")
        for index in index_split:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def get_ionization_aid(mol, acid_or_base=None):
    df_smarts_acid, df_smarts_base = split_acid_base_pattern()

    if mol == None:
        raise RuntimeError("read mol error: {}".format(mol_file))
    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)
    if acid_or_base == None:
        return acid_matches, base_matches
    elif acid_or_base == "acid":
        return acid_matches
    else:
        return base_matches
