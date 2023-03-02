---
title: "My Third Post"
date: 2023-03-02T20:41:40+01:00
draft: true
math: true
---

## Data <a class="anchor" id="first-bullet"></a>
Lets have a look at a few samples from our dataset


```python
names = [street.rstrip() for street in open("../data/streets_zh.txt")] #.rstrip() removes new line character '\n'
names[:5]
```




    ['Aargauerstrasse',
     'Abeggweg',
     'Abendweg',
     'Ackermannstrasse',
     'Ackersteinstrasse']



## Exact modelling - Count, count, count! <a class="anchor" id="second-bullet"></a>

As we mentioned earlier, we're going to build our street name model by looking at individual characters, specifically *pairs* of characters (so called *bigrams*). For our first model, we're going to count the number of times each bigram appears in our dataset.

Let's take a look at an example. The first street name in our dataset is 'Aargauerstrasse'. Let's split it into its bigrams


```python
for name in names[:1]:
    for pair in zip(name, name[1:]):
        print(pair)
```

    ('A', 'a')
    ('a', 'r')
    ('r', 'g')
    ('g', 'a')
    ('a', 'u')
    ('u', 'e')
    ('e', 'r')
    ('r', 's')
    ('s', 't')
    ('t', 'r')
    ('r', 'a')
    ('a', 's')
    ('s', 's')
    ('s', 'e')


In this case all our bigrams are unique, but if we iterate through all the street names and keep track of the bigrams we encounter, we find


```python
letters = set() # keep track of all the unique characters
pairs = set() # keep track of all the unique pairs 
count_pairs = dict() 

for name in names:
    for pair in zip(name, name[1:]):
        letters.add(pair[0])
        pairs.add(pair)
        count_pairs[pair] = 1 + count_pairs.get(pair, 0)     
```


```python
list(count_pairs.items())[:14] # print 14 first examples from our dictionary
```




    [(('A', 'a'), 1),
     (('a', 'r'), 112),
     (('r', 'g'), 129),
     (('g', 'a'), 169),
     (('a', 'u'), 109),
     (('u', 'e'), 25),
     (('e', 'r'), 629),
     (('r', 's'), 221),
     (('s', 't'), 1181),
     (('t', 'r'), 1110),
     (('r', 'a'), 1189),
     (('a', 's'), 1309),
     (('s', 's'), 1359),
     (('s', 'e'), 1366)]



And if we sort by the most common bigrams


```python
sorted(list(count_pairs.items()), key=lambda entry: -entry[1])[:14]
```




    [(('s', 'e'), 1366),
     (('s', 's'), 1359),
     (('a', 's'), 1309),
     (('r', 'a'), 1189),
     (('s', 't'), 1181),
     (('t', 'r'), 1110),
     (('e', 'r'), 629),
     (('e', 'n'), 584),
     (('e', 'g'), 462),
     (('w', 'e'), 367),
     (('c', 'h'), 358),
     (('n', 's'), 289),
     (('t', 'e'), 262),
     (('e', 'i'), 224)]



It's no surprise that the most common bigrams in our dataset are those that appear in the word 'strasse' (which means 'street' in German).

There is one thing we are missing though. We don't know how often a name starts with a given character nor when it ends with one. This would be useful to know as it will help us to create more realistic names.  We can solve this with a simple trick: add a unique special character, say `"$"`, at the beginning and end of every street name such that `'Aargauerstrasse'` becomes `'$Aargauerstrasse$'`. We now assign this special character the meaning of start and end of a word. Even though we are talking about Zurich, `"$"` is not occuring in any of the original street names so indeed we are free to assign it this meaning without it causing any confusion! Furthermore, we are free to use the same special character to mark beginning *and* end because its position within a bigram uniquely identifies whether it acts as a beginning- or end-marker


```python
for name in names[:1]:
    name = "$" + name + "$"
    for pair in zip(name, name[1:]):
        print(pair)
```

    ('$', 'A')
    ('A', 'a')
    ('a', 'r')
    ('r', 'g')
    ('g', 'a')
    ('a', 'u')
    ('u', 'e')
    ('e', 'r')
    ('r', 's')
    ('s', 't')
    ('t', 'r')
    ('r', 'a')
    ('a', 's')
    ('s', 's')
    ('s', 'e')
    ('e', '$')


Back to the bigram counts. To make things a bit more practical for us, let us express the counts as probabilities. For this we  restructure our dictionary `count_pairs` into a lookup table `freq_per_char` where every entry `freq_per_char[c_i][c_j]` corresponds to how often the character `c_i` was followed by the character `c_j`. We then can go from counts to probabilties by normalizing every row in this table such that it adds up to 1.


```python
# rerun the code from above with "$"s
names = ['$'+street.rstrip()+'$' for street in open("../data/streets_zh.txt")] #.rstrip() removes new line character '\n'
count_pairs = dict() 

for name in names:
    for pair in zip(name, name[1:]):
        letters.add(pair[0])
        pairs.add(pair)
        count_pairs[pair] = 1 + count_pairs.get(pair, 0)

# build our lookup table
freq_per_char = {c: dict() for c in letters}
for pair, frequency in count_pairs.items():
    first, second = pair
    freq_per_char[first][second] = frequency

# normalize rows
for k in freq_per_char.keys():
    total = sum(list(freq_per_char[k].values()))
    for v in freq_per_char[k]:
        freq_per_char[k][v]/= total
```

Lets visualize this lookup table


```python
import plotly.express as px
import pandas as pd

px.imshow(pd.DataFrame.from_dict(freq_per_char).T, width=1024, height=1024)
```
{{< rawhtml >}}
<div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.14.0.min.js"></script>                <div id="fb671a0e-1865-4f00-a714-6c58615a4a51" class="plotly-graph-div" style="height:800px; width:800px;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("fb671a0e-1865-4f00-a714-6c58615a4a51")) {                    Plotly.newPlot(                        "fb671a0e-1865-4f00-a714-6c58615a4a51",                        [{"coloraxis":"coloraxis","name":"0","x":["\u00f6","a","e","i","l","o","u","\u00fc","\u00e4","y","w","-","s","r","$","m","b","g","h","p","t","q","f","d","n","c",".","z","v","k"," ","x","j","F","K","M","R","B","G","H","L","O","S","W","E","N","P","A","Z","I","T","J","U","\u00e8","C","Q","D","V"],"y":["V","L","d","\u00f6","S","\u00e4","B","H","w","h","A","f","a","O","n","l","x"," ",".","b","e","q","i","t","Q","E","-","M","Z","m","R","r","c","N","J","g","s","I","C","y","j","u","F","P","v","T","o","$","\u00fc","K","p","\u00e8","k","z","D","W","U","G"],"z":[[0.10526315789473684,0.05263157894736842,0.2631578947368421,0.10526315789473684,0.05263157894736842,0.3157894736842105,0.10526315789473684,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.019230769230769232,0.2692307692307692,0.2692307692307692,0.15384615384615385,null,0.125,0.11538461538461539,0.009615384615384616,0.028846153846153848,0.009615384615384616,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.010101010101010102,0.04040404040404041,0.38636363636363635,0.06818181818181818,0.03787878787878788,0.03535353535353535,0.007575757575757576,null,0.007575757575757576,null,0.030303030303030304,0.05555555555555555,0.14393939393939395,0.012626262626262626,0.03535353535353535,0.025252525252525252,0.027777777777777776,0.020202020202020204,0.025252525252525252,0.005050505050505051,0.017676767676767676,0.0025252525252525255,0.0025252525252525255,0.0025252525252525255,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.010416666666666666,null,null,null,0.09375,null,null,null,null,null,0.020833333333333332,null,0.14583333333333334,0.125,null,0.020833333333333332,0.041666666666666664,0.041666666666666664,0.07291666666666667,0.020833333333333332,0.0625,null,0.020833333333333332,0.020833333333333332,0.19791666666666666,0.10416666666666667,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.03977272727272727,0.06534090909090909,0.06534090909090909,null,0.03409090909090909,0.019886363636363636,0.011363636363636364,0.008522727272727272,0.005681818181818182,null,null,null,null,null,null,null,null,null,0.07102272727272728,0.4034090909090909,null,null,null,null,0.2755681818181818,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,0.034482758620689655,null,0.08045977011494253,null,null,null,null,null,0.10344827586206896,0.14942528735632185,null,0.034482758620689655,0.022988505747126436,0.08045977011494253,0.05747126436781609,0.022988505747126436,0.04597701149425287,null,0.034482758620689655,0.05747126436781609,0.11494252873563218,0.16091954022988506,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.019417475728155338,0.19902912621359223,0.16019417475728157,0.10194174757281553,0.06310679611650485,0.07281553398058252,0.1262135922330097,0.05825242718446602,0.04854368932038835,null,null,null,null,0.15048543689320387,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.06763285024154589,0.2028985507246377,0.2753623188405797,0.10144927536231885,null,0.21256038647342995,0.06280193236714976,0.043478260869565216,0.028985507246376812,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.004830917874396135,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.04585152838427948,0.8013100436681223,0.14192139737991266,null,null,null,null,0.002183406113537118,0.006550218340611353,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.002183406113537118,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.02455661664392906,0.13915416098226466,0.1432469304229195,0.03547066848567531,0.12551159618008187,0.1377899045020464,0.02455661664392906,0.023192360163710776,0.009549795361527967,0.001364256480218281,0.05184174624829468,0.01227830832196453,0.09004092769440655,0.03137789904502047,0.005457025920873124,0.021828103683492497,0.020463847203274217,0.008185538881309686,0.013642564802182811,0.002728512960436562,0.0286493860845839,null,0.005457025920873124,0.002728512960436562,0.03547066848567531,null,null,0.004092769440654843,0.001364256480218281,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.00847457627118644,0.03389830508474576,null,0.2542372881355932,null,0.1440677966101695,null,null,null,null,0.00847457627118644,0.03389830508474576,0.0847457627118644,null,0.1694915254237288,0.025423728813559324,0.01694915254237288,0.00847457627118644,0.03389830508474576,0.00847457627118644,null,0.00847457627118644,0.01694915254237288,0.1016949152542373,0.025423728813559324,null,0.00847457627118644,null,0.00847457627118644,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.047619047619047616,0.22943722943722944,0.03896103896103896,0.047619047619047616,0.008658008658008658,0.030303030303030304,null,0.004329004329004329,null,0.03896103896103896,0.030303030303030304,0.16017316017316016,0.04329004329004329,0.09956709956709957,0.004329004329004329,0.021645021645021644,0.025974025974025976,0.017316017316017316,0.017316017316017316,0.012987012987012988,0.004329004329004329,0.09523809523809523,null,0.012987012987012988,null,null,null,null,null,0.008658008658008658,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.0030594405594405595,0.0013111888111888112,0.021853146853146852,0.05638111888111888,null,0.047639860139860137,null,null,null,0.0026223776223776225,0.004370629370629371,0.5721153846153846,0.04895104895104895,null,0.0118006993006993,0.008741258741258742,0.010926573426573426,0.006118881118881119,0.005244755244755245,0.055944055944055944,null,0.004370629370629371,0.015297202797202798,0.062062937062937064,0.052884615384615384,null,0.0017482517482517483,0.0030594405594405595,0.0021853146853146855,null,0.0013111888111888112,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,0.13953488372093023,null,0.06976744186046512,null,null,null,null,null,0.023255813953488372,null,0.046511627906976744,0.023255813953488372,null,null,0.27906976744186046,null,0.023255813953488372,0.06976744186046512,0.3023255813953488,null,null,null,null,null,null,null,null,0.023255813953488372,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.04460093896713615,0.10093896713615023,0.018779342723004695,0.004694835680751174,0.006259780907668232,0.002347417840375587,0.000782472613458529,null,null,0.0837245696400626,0.02034428794992175,0.22613458528951486,0.024256651017214397,0.032081377151799685,0.00782472613458529,0.050078247261345854,0.11267605633802817,0.04460093896713615,0.009389671361502348,0.033646322378716745,0.000782472613458529,0.009389671361502348,0.05086071987480438,0.04929577464788732,0.001564945226917058,null,0.027386541471048513,null,0.023474178403755867,0.013302034428794992,null,0.000782472613458529,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.005623242736644799,0.1190253045923149,0.14058106841611998,0.1668228678537957,0.07122774133083412,0.02530459231490159,0.01499531396438613,0.006560449859418931,0.006560449859418931,null,0.030927835051546393,0.022492970946579195,0.09653233364573571,0.004686035613870665,0.007497656982193065,0.007497656982193065,0.026241799437675725,0.02717900656044986,0.016869728209934397,0.00937207122774133,0.022492970946579195,0.0009372071227741331,0.01780693533270853,0.08903467666354264,0.0009372071227741331,0.011246485473289597,null,0.04217432052483599,0.0028116213683223993,0.006560449859418931,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null,0.2,0.8,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,0.008928571428571428,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.008928571428571428,0.09821428571428571,null,null,null,null,null,null,null,null,null,0.008928571428571428,0.0625,0.044642857142857144,0.05357142857142857,0.09821428571428571,0.0625,0.125,0.026785714285714284,0.008928571428571428,0.13392857142857142,0.08928571428571429,0.026785714285714284,0.008928571428571428,0.026785714285714284,0.026785714285714284,0.026785714285714284,0.008928571428571428,0.026785714285714284,0.008928571428571428,0.008928571428571428,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null,null,0.3,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.7,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.15598885793871867,0.4596100278551532,0.03899721448467967,0.036211699164345405,0.013927576601671309,0.033426183844011144,0.0807799442896936,0.005571030640668524,0.002785515320334262,null,0.011142061281337047,0.04456824512534819,0.08356545961002786,0.002785515320334262,0.002785515320334262,0.005571030640668524,0.005571030640668524,0.013927576601671309,null,0.002785515320334262,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.001544799176107106,0.008496395468589083,0.057672502574665295,0.05690010298661174,0.0020597322348094747,0.011071060762100926,null,null,0.0012873326467559218,0.0012873326467559218,0.004119464469618949,0.03141091658084449,0.16194644696189495,0.3303295571575695,0.004119464469618949,0.009783728115345005,0.11894953656024716,0.007209062821833162,0.0025746652935118436,0.01416065911431514,0.00025746652935118434,0.0028321318228630276,0.012100926879505664,0.15036045314109167,0.004634397528321318,null,0.0005149330587023687,0.00025746652935118434,null,0.004119464469618949,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,1.0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.02187784867821331,0.15587967183226983,null,0.041932543299908843,0.0054694621695533276,0.0036463081130355514,null,0.0009115770282588879,null,0.03919781221513218,0.019143117593436645,0.13855970829535097,0.03646308113035551,0.027347310847766638,0.0300820419325433,0.01731996353691887,0.09571558796718323,0.02187784867821331,0.007292616226071103,0.03828623518687329,null,0.004557885141294439,0.03372835004557885,0.18778486782133091,0.04102096627164995,null,0.0027347310847766638,0.0027347310847766638,0.024612579762989972,null,0.0009115770282588879,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.0009115770282588879,null,null,null,null],[0.0025933609958506223,0.027489626556016597,0.1358921161825726,0.01970954356846473,0.014004149377593362,0.016078838174273857,0.0031120331950207467,0.002074688796680498,0.0015560165975103733,null,0.014004149377593362,0.013485477178423237,0.03371369294605809,0.5757261410788381,0.007261410788381743,null,0.004149377593360996,0.00466804979253112,0.014522821576763486,0.001037344398340249,0.0487551867219917,0.0005186721991701245,0.0005186721991701245,null,0.002074688796680498,null,0.0036307053941908715,0.05342323651452282,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,1.0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,0.22727272727272727,0.10227272727272728,null,0.045454545454545456,null,null,0.022727272727272728,null,null,0.07954545454545454,0.14772727272727273,null,0.045454545454545456,0.011363636363636364,0.06818181818181818,null,null,0.045454545454545456,null,0.011363636363636364,0.045454545454545456,0.125,null,null,null,null,0.022727272727272728,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.007905138339920948,null,null,null,null,0.05533596837944664,0.02766798418972332,0.015810276679841896,0.03162055335968379,0.05138339920948617,0.039525691699604744,0.07509881422924901,0.015810276679841896,0.011857707509881422,0.30039525691699603,0.20553359683794467,0.007905138339920948,0.011857707509881422,0.05928853754940711,0.007905138339920948,0.011857707509881422,0.007905138339920948,0.003952569169960474,0.007905138339920948,null,null,0.023715415019762844,0.003952569169960474,0.007905138339920948,0.007905138339920948],[0.03636363636363636,0.4,0.11818181818181818,0.11818181818181818,null,0.12727272727272726,0.045454545454545456,0.14545454545454545,null,0.00909090909090909,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.03773584905660377,0.03773584905660377,0.33962264150943394,0.1509433962264151,null,0.03773584905660377,0.05660377358490566,0.05660377358490566,0.07547169811320754,0.018867924528301886,0.16981132075471697,null,0.018867924528301886,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.1986970684039088,0.15960912052117263,0.048859934853420196,0.016286644951140065,0.029315960912052116,0.003257328990228013,0.035830618892508145,null,null,0.009771986970684038,0.019543973941368076,0.07166123778501629,null,null,0.0749185667752443,0.04234527687296417,0.019543973941368076,0.009771986970684038,0.05537459283387622,0.013029315960912053,null,0.006514657980456026,null,null,null,null,null,null,null,0.18566775244299674,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.09022556390977443,0.09022556390977443,0.24812030075187969,0.18796992481203006,null,0.19548872180451127,0.07518796992481203,0.08270676691729323,0.022556390977443608,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.007518796992481203,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.0016522098306484924,0.49111937216026436,0.05163155720776538,0.04626187525815779,0.015282940933498555,0.016522098306484923,0.011152416356877323,0.01321767864518794,0.003304419661296985,0.0004130524576621231,0.027674514663362248,0.018174308137133416,0.0912845931433292,0.008674101610904586,0.011152416356877323,0.007021891780256092,0.010326311441553077,0.05328376703841388,0.009913258983890954,0.0057827344072697235,0.027674514663362248,0.0004130524576621231,0.00784799669558034,0.021065675340768277,0.0214787277984304,0.009087154068566708,null,0.0053696819496076,0.0012391573729863693,0.00660883932259397,0.0053696819496076,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.004291845493562232,0.002145922746781116,0.004291845493562232,null,0.004291845493562232,null,null,null,null,null,null,null,null,null,null,null,null,0.7682403433476395,null,null,null,null,null,null,0.002145922746781116,null,null,null,0.2145922746781116,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.022222222222222223,0.13333333333333333,0.4222222222222222,0.1111111111111111,null,0.15555555555555556,0.044444444444444446,0.044444444444444446,0.06666666666666667,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.24,0.04,null,null,0.36,0.32,null,0.04,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.15490375802016498,0.10999083409715857,0.01924839596700275,0.016498625114573784,0.00458295142071494,0.01008249312557287,0.0018331805682859762,0.0018331805682859762,null,0.031164069660861594,0.006416131989000917,0.08890925756186985,0.017415215398716773,0.43354720439963335,0.0036663611365719525,0.0018331805682859762,0.06416131989000917,0.012832263978001834,0.006416131989000917,0.0018331805682859762,null,0.002749770852428964,0.0018331805682859762,0.007332722273143905,null,null,null,null,0.0009165902841429881,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.0033629594042757626,0.3281287533029066,0.0055248618784530384,0.006725918808551525,0.0009608455440787893,null,null,0.00048042277203939464,0.00024021138601969732,0.006725918808551525,0.0024021138601969735,0.3264472736007687,0.0009608455440787893,0.002882536632236368,0.0016814797021378813,0.006005284650492433,0.002882536632236368,0.0019216910881575786,0.0019216910881575786,0.28368964688926257,0.00024021138601969732,0.00024021138601969732,0.0009608455440787893,0.00024021138601969732,0.015133317319240933,null,null,null,0.00024021138601969732,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,0.04285714285714286,null,null,null,null,null,null,null,0.014285714285714285,0.04285714285714286,null,0.6857142857142857,null,null,null,null,null,null,null,0.02857142857142857,0.17142857142857143,null,null,null,null,null,null,null,null,0.014285714285714285,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.25,0.0625,null,0.125,0.1875,0.0625,null,0.0625,0.03125,null,null,null,0.03125,null,null,null,null,0.15625,null,null,null,null,null,null,null,0.03125,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.03125,0.15625,null,0.125,null,null,null,null,null,null,null,0.1875,0.09375,0.03125,0.03125,0.03125,null,0.03125,0.03125,0.03125,null,null,0.125,0.03125,0.03125,null,null,null,0.03125,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.5,0.25,null,null,0.25,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.02654867256637168,0.05530973451327434,0.00663716814159292,0.07079646017699115,null,null,null,0.0022123893805309734,0.0022123893805309734,0.015486725663716814,null,0.1902654867256637,0.1261061946902655,0.00663716814159292,0.07300884955752213,0.05309734513274336,0.059734513274336286,0.01991150442477876,0.01327433628318584,0.05309734513274336,null,0.028761061946902654,0.028761061946902654,0.08849557522123894,0.05752212389380531,null,0.01327433628318584,null,0.0022123893805309734,null,0.0022123893805309734,0.004424778761061947,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.018018018018018018,0.06306306306306306,0.21621621621621623,0.04504504504504504,0.08108108108108109,0.07207207207207207,0.06306306306306306,0.02702702702702703,0.009009009009009009,null,null,null,null,0.38738738738738737,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.009009009009009009,null,0.009009009009009009,null,null,null,null,null,null,null,null],[null,0.22413793103448276,0.1206896551724138,0.05172413793103448,0.2413793103448276,0.017241379310344827,null,0.017241379310344827,null,null,null,null,null,0.15517241379310345,null,null,null,null,0.034482758620689655,null,null,null,0.13793103448275862,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.13636363636363635,0.45454545454545453,0.13636363636363635,null,0.09090909090909091,0.045454545454545456,null,null,null,null,0.09090909090909091,null,0.045454545454545456,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.04918032786885246,0.19672131147540983,0.06557377049180328,0.08196721311475409,null,0.13114754098360656,0.16393442622950818,0.01639344262295082,null,null,null,null,null,0.19672131147540983,null,null,null,null,0.09836065573770492,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.0019011406844106464,0.0038022813688212928,null,0.1596958174904943,0.04752851711026616,0.0076045627376425855,null,null,0.0019011406844106464,0.009505703422053232,0.011406844106463879,0.10456273764258556,0.14638783269961977,null,0.03231939163498099,0.055133079847908745,0.024714828897338403,0.04182509505703422,0.019011406844106463,0.043726235741444866,0.0019011406844106464,0.13688212927756654,0.015209125475285171,0.09505703422053231,0.026615969581749048,null,0.0019011406844106464,0.0038022813688212928,0.005703422053231939,null,null,0.0019011406844106464,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.04731075697211155,0.061752988047808766,0.05029880478087649,0.05926294820717132,0.09063745019920319,0.060756972111553786,0.08665338645418327,0.04830677290836653,0.01942231075697211,0.1299800796812749,0.05776892430278884,0.04133466135458167,0.02041832669322709,0.0199203187250996,0.055776892430278883,0.023406374501992032,0.03286852589641434,0.02838645418326693,0.010956175298804782,0.009462151394422311,null,0.012948207171314742,0.00149402390438247,0.022410358565737053,0.00846613545816733],[null,null,0.011049723756906077,null,0.016574585635359115,null,null,null,null,null,null,null,0.08287292817679558,0.12154696132596685,null,0.011049723756906077,0.027624309392265192,0.03867403314917127,0.292817679558011,0.0055248618784530384,0.14917127071823205,null,0.0055248618784530384,0.03314917127071823,0.11602209944751381,0.08839779005524862,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.043478260869565216,0.2246376811594203,0.07971014492753623,0.10144927536231885,0.15217391304347827,0.09420289855072464,0.057971014492753624,0.057971014492753624,0.043478260869565216,0.007246376811594203,null,null,null,0.13043478260869565,null,null,null,null,null,null,null,null,null,null,0.007246376811594203,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.005649717514124294,0.062146892655367235,0.12429378531073447,0.1016949152542373,0.3220338983050847,0.02824858757062147,0.005649717514124294,null,null,0.011299435028248588,0.005649717514124294,null,0.02824858757062147,0.06779661016949153,null,null,null,null,0.03954802259887006,0.06779661016949153,0.011299435028248588,null,0.1016949152542373,null,null,null,null,null,null,0.011299435028248588,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.005649717514124294,null,null,null,null],[null,null,null,null,null,null,null,null,null,null,null,null,null,0.5,null,null,null,null,null,null,null,null,null,null,null,0.5,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.009950248756218905,0.06965174129353234,0.5422885572139303,0.014925373134328358,0.07462686567164178,0.05472636815920398,0.014925373134328358,null,null,null,0.014925373134328358,null,0.05472636815920398,0.004975124378109453,0.024875621890547265,0.004975124378109453,0.004975124378109453,0.01990049751243781,0.029850746268656716,null,0.03980099502487562,null,null,null,null,null,null,0.004975124378109453,null,0.014925373134328358,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.013761467889908258,0.21100917431192662,0.05504587155963303,0.04128440366972477,0.022935779816513763,0.009174311926605505,null,null,null,0.07339449541284404,0.022935779816513763,0.13761467889908258,0.009174311926605505,0.3165137614678899,0.009174311926605505,0.009174311926605505,0.03669724770642202,0.009174311926605505,0.013761467889908258,null,null,null,null,0.0045871559633027525,null,null,0.0045871559633027525,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[0.06382978723404255,0.19148936170212766,0.14893617021276595,0.1276595744680851,null,0.1702127659574468,0.10638297872340426,0.02127659574468085,null,null,null,null,null,0.1702127659574468,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.20224719101123595,0.47752808988764045,0.17415730337078653,null,0.06179775280898876,0.011235955056179775,0.02247191011235955,0.011235955056179775,0.03932584269662921,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,null,0.25,null,0.15,null,null,null,null,null,null,null,0.05,0.1,null,null,null,null,0.05,null,0.05,null,null,null,0.35,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],[null,0.09352517985611511,0.17985611510791366,0.08633093525179857,0.1223021582733813,0.07913669064748201,0.12949640287769784,0.014388489208633094,0.03597122302158273,null,0.007194244604316547,null,0.007194244604316547,0.2158273381294964,null,0.007194244604316547,null,null,null,null,null,null,0.007194244604316547,null,0.014388489208633094,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]],"type":"heatmap","xaxis":"x","yaxis":"y","hovertemplate":"x: %{x}<br>y: %{y}<br>color: %{z}<extra></extra>"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"scaleanchor":"y","constrain":"domain"},"yaxis":{"anchor":"x","domain":[0.0,1.0],"autorange":"reversed","constrain":"domain"},"coloraxis":{"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"margin":{"t":60},"height":760,"width":720},                        {"responsive": true}                    )                };                            </script>        </div>
{{< /rawhtml >}}

This gives us a nice and pretty overview! Also an opportunity for us to check if things look like we expect them to do. For example, we see that the `"$"` is only followed by capital letters, which makes sense. 

We are now ready to generate new street names 🥁! We can do this by iteratively adding new characters to the starting string `"$"`. At each step, we'll use the last character we added to the string to determine the probability distribution of the next character via our lookup table. We'll keep adding new characters until we reach the "$" character again, giving us a new Zurich street name! Let's see how this plays out...


```python
from random import choices
n_samples = 20

for _ in range(n_samples):
    gen = ["$"]
    while True:
        c = gen[-1]
        samples = list(freq_per_char[c].keys())
        p_samples = list(freq_per_char[c].values())
        next_c = choices(samples, p_samples)[0]
        gen.append(next_c)
        if next_c=="$":
            break
    print("".join(gen))
```

    $Enserassenrassssere$
    $Sche$
    $Döhtrtrsengasefe$
    $Serstsig$
    $Hom Ulenswigf$
    $Krase Am-Miebe$
    $Sinlsstramaleneg-Schl-Heg$
    $Tenstr He$
    $Füdedoregase$
    $Spestse$
    $Grinstitrastrul-Sthasseiste$
    $Dimpenstrbe$
    $Sigensbennenbänardste$
    $Ne$
    $Okwerarasseng$
    $Sck$
    $Lirasat-Wossenschaserkeg$
    $Gubransegastrlde$
    $Futofwin-We$
    $Hützisstradegwebe$


I hope you are not too disappointed😔. The results indeed seem a bit underwhelming ... However, lets look at what the names would look like if we would sample every next letter with equal probability


```python
from random import choices
n_samples = 20

for _ in range(n_samples):
    gen = ["$"]
    while True:
        c = gen[-1]
        #samples = list(freq_per_char[c].keys())
        #p_samples = list(freq_per_char[c].values())
        #next_c = choices(samples, p_samples)[0]
        next_c = choices(list(letters))[0]
        gen.append(next_c)
        if next_c=="$":
            break
    print("".join(gen))
```

    $c.KnhKk VTwdnszF$
    $FWawZ NebWèWy KlytebiGqfdONüy C-OhSjiafPMhnepädK jbTwmfüiRxvFMyRRDNMeNa.bufKetIèdTKZOSbrGKsHQq-ypuUBjvspNR aBaqGfTy-ügyFLvvNmüa$
    $äDZ-xtwrQEVoUdTQfDwèBJ QQsZR.jUnSylVeMlW- e-myTjeLoxala$
    $AixaMCfAäG.ägydöxkCOPWUUücaEfwJ$
    $iäSgznvB bènNkHElgH-sjäcü$
    $EFTyTBtNpSSSxIeyartQWewbmLjyüZJCi.vtPMBvriMxKwxZVuMZHrJMEzdyzoTyEMkuUC äGWz$
    $orfQxaRiCyhscèNAtKxsOjdisfskhcuyWVCOkpörpVqxeUrDLvfQSC$
    $oLMyddVlNCüTfwüvjP h-BEpOuEU$
    $vnpn.pOpönBtqöRCü.rqm$
    $w$
    $rW iQxSFBdnFISWVèNug$
    $ZF$
    $BoEyRDTUCmMIU$
    $IèHmAfNtHFkväCLqP-DUSUVmSWfF-shxrvècömi$
    $.dpuPDdGFBfPGrzn-yRpcWnAZWscUrd.bAwoDf fEkZctIediUTNgWDG hZrUMrrcHOLDVBnDQMiOgüDgE.BCZaSèluwIraqsIqIUiuaJgKVEüvabGFhdgOQqqajrFWFky-özfKAcwhsJüLFZèhèfaFSmCIOiüAcDm$
    $nlBI$
    $fbdyGPjNHRUV$
    $VL$
    $WTwgDDsKfsCrEE.kè HFFQT$
    $uHuBQGicvhPKx.GzJ tQJGtGyGzöHvfwüc t.e$


Uiii!!! So we see that while our bigram model is not able to come up with super realistic street names, they are still way closer to real street names than these.

Why is our bigram model not performing better though? Turns out that only looking at the previous character to predict the next character is not enough to capture the "nature" of street names. We will extend our bigram model to n-gram models (i.e. look at the previous $n-1$ characters to predict the next character) a bit later. For now, lets stick with bigrams. 

As announced, in the next part we are going to build the same model using a simple neural network. Before we do that, however, we will need to clarify what we mean by "same" model. How can we compare different models if they have been constructed in totally different ways? It turns out that looking at the probabilities for every pair and multiplying them together is a measure of how well the model predicts the dataset. In the next section we explore and motivate this in more detail, however, feel free to skip it if you are not interested in the technical details.
