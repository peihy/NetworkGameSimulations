from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import networkx as nx
from operator import itemgetter
import time
import math

'''
Initial setup

rounds: Number of rounds per game
players: Network size
frac: Fraction of rewired edges
ncolors: number of colors
shrt: number of shortcuts (not in use anymore)
color_names: "names" of the colors
drawcolors: colorcodes for plotting purposes
gametype: Determines the progress and end goal (only Consensus in this file, for graph coloring check simugc.py)
colors: playercolors listed

'''
rounds = 50
players = 24
frac = 0.15
ncolors = 3
shrt = 7
color_names = [1, 2, 3]
drawcolors = ['r', 'g', 'b']
gametype = "Consensus" #UNUSED, GC IN DIFFERENT FILE

colors=[]
for i in range(players):
    colors.append(color_names[i%ncolors])
#random.shuffle(colors)
def initialforms():
    '''
    Function for testing different topologies
    !Not used in games!
    '''

    B = nx.cycle_graph(players)
    B = nx.convert_node_labels_to_integers(B, first_label = 1, ordering="sorted")
    b_pos = nx.circular_layout(B)
    for i in range(1,players+1):
        if i < players-3:
            B.add_edge(i,i+4)
        elif i == players:
            B.add_edge(i, 4)
        elif i == players-1:
            B.add_edge(i, 3)
        elif i == players-2:
            B.add_edge(i, 2)
        elif i == players-3:
            B.add_edge(i, 1)
        else:
            B.add_edge(i, players-i-1)
    plt.figure("Initial regular")
    z=1
    for color in colors:
        nx.draw_networkx_nodes(B, b_pos, nodelist=[z], node_color=drawcolors[color-1])
        z+=1
    nx.draw_networkx_edges(B,b_pos,width=1.0,alpha=0.5)
    nx.draw_networkx_labels(B,b_pos,width=1.0,alpha=0.5)

    #A = nx.watts_strogatz_graph(players, 4, 0.3, seed=None)
    A = nx.cycle_graph(players)
    A = nx.convert_node_labels_to_integers(A, first_label = 1, ordering="sorted")
    a_pos = nx.circular_layout(A)
    shorts1, shorts2, x = [], [], 1
    for i in range(shrt):
        s = random.randrange(3,players)
        if s not in shorts1 or s not in shorts2:
            shorts1.append(s)
            shorts2.append(s-2)
    x=0
    random.shuffle(shorts1)
    random.shuffle(shorts2)
    for i in range(1,players+1):
        if i not in shorts2:
            if i < players-1:
                A.add_edge(i,i+2)
            elif i == players:
                A.add_edge(i, 2)
            else:
                A.add_edge(i, players-i)
    A.add_edge(shorts1[0], shorts2[len(shorts2)-1])
    for i in range(len(shorts2)-1):
        A.add_edge(shorts2[i], int(shorts1[i+1]))

    while nx.average_degree_connectivity(A)[4]!=4:
        A = nx.cycle_graph(players)
        A = nx.convert_node_labels_to_integers(A, first_label = 1, ordering="sorted")
        a_pos = nx.circular_layout(A)
        shorts1, shorts2, x = [], [], 1
        links = math.ceil(players*frac)
        x=0
        nlink = 0
        rew1,rew2=[],[]

        for i in range(1,players+1):
            if i not in shorts2:
                if i < players-1:
                    A.add_edge(i,i+2)
                elif i == players:
                    A.add_edge(i, 2)
                else:
                    A.add_edge(i, players-i)

        while nlink < links:
            rewired = random.randint(1,players)
            tmp=[]
            for n in nx.all_neighbors(A, rewired):
                tmp.append(n)
            neib = random.choice(tmp)
            rew1.append(rewired)
            rew2.append(neib)
            A.remove_edge(rewired, neib)
            nlink +=1
        random.shuffle(rew2)
        random.shuffle(rew1)
        for r in range(len(rew1)):
            if rew1[r]!=rew2[r]:
                A.add_edge(rew1[r], rew2[r])
    plt.figure("Initial smallworld")
    z=1
    for color in colors:
        nx.draw_networkx_nodes(A, a_pos, nodelist=[z], node_color=drawcolors[color-1])
        z+=1
    nx.draw_networkx_edges(A,a_pos,width=1.0,alpha=0.5)
    nx.draw_networkx_labels(A,a_pos,width=1.0,alpha=0.5)

    bls=64
    d1=players-1
    tmp=[]
    while d1!=0:
        if players%d1 ==0:
            tmp.append(d1)
        d1-=1
    middle = (float(len(tmp))/2)
    if middle % 2 != 0:
        d1=tmp[int(middle - .5)]
        d2=int(players/d1)
    else:
        d1=tmp[int(middle-1)]
        d2=int(players/d1)
    fcolors=[]
    z=0
    for i in range(players):
        fcolors.append(color_names[(i+z)%ncolors])
        if (i+1)%d2 == 0:
            z+=1
        if z == ncolors-1 and d1%3!=0:
            z=0
    links = math.ceil(players*frac)
    F = nx.grid_graph(dim=[int(d1), int(d2)])
    while nx.average_degree_connectivity(F)[4]!=4:
        nlink = 0
        rew1,rew2=[],[]
        F = nx.grid_graph(dim=[int(d1), int(d2)])
        F = nx.convert_node_labels_to_integers(F, first_label = 1, ordering="sorted")

        square = d1
        pos, i, row = {}, 1, 0
        for y in range(d1):
            if y+1 <= d1/2:
                for x in range(d2):
                    if x+1 <= d2/2:
                        pos[i] = [-x*d2+((d1-y-1)-(d1/2))**2, (-y*d1+((d2-x)-(d2/2))**2)*0.5]
                    elif x+1 > d2/2:
                        pos[i] = [-x*d2-((d1-y-1)-(d1/2))**2, (-y*d1+(x+1-(d2/2))**2)*0.5]
                    else:
                        pos[i] = [(-(d2-1)*d2)/2, y*(-(d1-1)*d1)/(d1-1)]
                    i +=1
            if y+1 > d1/2:
                for x in range(d2):
                    if x+1 <= d2/2:
                        pos[i] = [(-x)*d2+(y-(d1/2))**2, (-y*d1-((d2-x)-(d2/2))**2)*0.5]
                    elif x+1 > d2/2:
                        pos[i] = [-x*d2-(y-(d1/2))**2, (-y*d1-(x+1-(d2/2))**2)*0.5]
                    else:
                        pos[i] = [(-(d2-1)*d2)/2, y*(-(d1-1)*d1)/(d1-1)]
                    i +=1

        bot,top,left,right=[],[],[],[]
        for j in range(1, players+1):
            if j <=d2:
                bot.append(j)
            if j>(players-d2):
                top.append(j)
            if (j-1)%(d2)==0:
                right.append(j)
            if j%(d2)==0:
                left.append(j)
        Cr = nx.average_clustering(F)
        if False:
            random.shuffle(right)
            random.shuffle(left)
            random.shuffle(bot)
            random.shuffle(top)
        for r in range(len(right)):
            F.add_edge(left[r], right[r])
        for r in range(len(top)):
            F.add_edge(top[r], bot[r])
        if True:
            while nlink < links:
                rewired = random.randint(1,players)
                tmp=[]
                for n in nx.all_neighbors(F, rewired):
                    tmp.append(n)
                neib = random.choice(tmp)
                rew1.append(rewired)
                rew2.append(neib)
                F.remove_edge(rewired, neib)
                nlink +=1
            random.shuffle(rew2)
            random.shuffle(rew1)
            for r in range(len(rew1)):
                if rew1[r]!=rew2[r]:
                    F.add_edge(rew1[r], rew2[r])
    plt.figure("Initial mesh")
    z=1
    for color in fcolors:
        nx.draw_networkx_nodes(F, pos, nodelist=[z], node_color=drawcolors[color-1])
        z+=1
    nx.draw_networkx_edges(F,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_labels(F,pos,width=1.0,alpha=0.5)
    #print(nx.average_shortest_path_length(B))
    #print(nx.average_shortest_path_length(A))
    print(nx.average_shortest_path_length(F))
    print(nx.average_clustering(F))
    Rg = nx.random_regular_graph(4, players)
    #print((nx.average_clustering(F)/Cr)-(nx.average_shortest_path_length(F)/nx.average_shortest_path_length(Rg)))
    #print(nx.average_degree_connectivity(F))

#initialforms()

'''
Initial setup ends
'''
#Create the players, nodes and networkx
class Player(object):
    def __init__(self):
        self.id = 0
        self.group = None
        self.place = 0
        self.selection = 0
        self.requested = False
        self.accepted = 0
        self.color = 0
        self.bot_person = 0
        self.cluster_size = 1

    #Handle the different phases for each player type
    def accept(self):
        if self.bot_person == 1:
            self.greedy("acc")
        else:
            self.noisy("acc")

    def request(self):
        if self.bot_person == 1:
            self.greedy("req")
        else:
            self.noisy("req")


    def noisy(self, string):
        smart=0
        if string == "acc":
            requestors = []
            for n in nx.all_neighbors(self.group.network, self.id):
                neib = self.group.get_player_by_id(n)
                if neib.color != self.color and neib.selection == self.id:
                    requestors.append(neib.id)
            if len(requestors) > 0:
                self.accepted = random.choice(requestors)
            self.requested = False
        else:
            choice = random.randint(1,100)
            requestables = []
            if choice <= self.group.noisyness:
                for n in nx.all_neighbors(self.group.network, self.id):
                    neib = self.group.get_player_by_id(n)
                    if neib.color != self.color:
                        requestables.append(neib)
                if len(requestables) > 0:
                    self.selection = random.choice(requestables).id
                    self.group.get_player_by_id(self.selection).requested = True
            else:
                cs = 10000
                for n in nx.all_neighbors(self.group.network, self.id):
                    neib = self.group.get_player_by_id(n)
                    if neib.color != self.color and neib.cluster_size <= cs:
                        if cs == neib.cluster_size:
                            requestables.append(neib)
                        else:
                            cs = neib.cluster_size
                            requestables = []
                            requestables.append(neib)
                if len(requestables) != 0:
                    self.selection = random.choice(requestables).id
                    self.group.get_player_by_id(self.selection).requested = True


    def greedy(self, string):
        '''
        The main real bot logic
        Player requests the neighbor with the lowest cluster size if he is not in the "largest" cluster and the noise does not proc
        Same goes for accepting
        '''
        if self.group.info_vis:
            smart = self.smart_req()
        else: smart = 0
        if string == "acc":
            requestors = []
            for n in nx.all_neighbors(self.group.network, self.id):
                neib = self.group.get_player_by_id(n)
                if neib.color != self.color and neib.selection == self.id:
                    requestors.append(neib.id)
            if len(requestors) > 0:
                if self.in_largest() and self.group.info_clust:
                    for i in requestors:
                        if self.common_neighbour(i):
                            self.accepted = i
                else:
                    self.accepted = random.choice(requestors)
                    for i in requestors:
                        if self.common_neighbour(i):
                            self.accepted = i
                    if smart != 0:
                        tmp = self.cluster_size
                        random.shuffle(requestors)
                        for r in requestors:
                            if smart[r]>tmp:
                                tmp=smart[r]
                                self.accepted = r
            self.requested = False
        else:
            requestables = []
            cs = 100000
            if self.in_largest() and self.group.info_clust:
                return
            for n in nx.all_neighbors(self.group.network, self.id):
                neib = self.group.get_player_by_id(n)
                if neib.color != self.color and neib.cluster_size <= cs:
                    if cs == neib.cluster_size:
                        requestables.append(neib)
                    else:
                        cs = neib.cluster_size
                        requestables = []
                        requestables.append(neib)
            if smart != 0 and len(requestables) != 0:
                tmp = self.cluster_size
                random.shuffle(requestables)
                for r in requestables:
                    if smart[r.id]>tmp:
                        tmp=smart[r.id]
                        self.selection = r.id
                self.group.get_player_by_id(self.selection).requested = True
            else:
                if len(requestables) != 0:
                    self.selection = random.choice(requestables).id
                    self.group.get_player_by_id(self.selection).requested = True


    def common_neighbour(self, req):
        '''
        Eases the passing by of large clusters
        '''
        #IS THIS ACCEPTABLE
        return False
        for n in nx.all_neighbors(self.group.network, self.id):
            neib = self.group.get_player_by_id(n)
            for n2 in nx.all_neighbors(self.group.network, req):
                neibneib = self.group.get_player_by_id(n2)
                if neib.id == neibneib.id:
                    if neibneib.color == self.color:
                        return True
        if self.cluster_size == 1:
            return True
        return False

    def smart_req(self):
        '''
        PROSPECTIVE CLUSTERSIZE
        '''
        choice = {}
        for n in nx.all_neighbors(self.group.network, self.id):
            neib = self.group.get_player_by_id(n)
            prosp = 1
            if neib.color != self.color:
                for n2 in nx.all_neighbors(self.group.network, neib.id):
                    neibneib = self.group.get_player_by_id(n2)
                    if self.id != neibneib.id and neibneib.color == self.color:
                        if neibneib.cluster_size + 1 > prosp:
                            prosp = neibneib.cluster_size +1
            choice[neib.id] = prosp
        if len(choice)>0:
            return choice
        return 0


    def in_largest(self):
        '''
        Global knowledge into decision to stay or to swap
        '''
        if random.randrange(100) < self.group.noisyness:
            return False
        tmp = []
        for p in self.group.players:
            if self.color == p.color: tmp.append(p.cluster_size)
        largest = max(tmp)
        full_clust = len(tmp)
        #if self.cluster_size == largest:
        prob = self.cluster_size/full_clust
        if prob > self.group.threshold:
            return True
        #if random.randrange(100) <= (prob**2)*100:
        #    return True
        return False

    def player_print(self):
        print(self.id, self.color, self.place, self.bot_person)

class Group(object):
    def __init__(self, players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle):
        self.mapper = []
        self.players = []
        self.noise = noise
        self.noisyness = noisyness
        self.info_vis = info_vis
        self.info_clust = info_clust
        self.threshold = threshold
        self.noisybots = []
        for x in range(players):
            #self.noisybots.append(random.randint(1,players))
            if noise: self.noisybots.append(x+1)

        #print(self.noisybots)
        '''
        SHUFFLE
        '''
        self.colors=[]
        if net != "C":
            for i in range(players):
                self.colors.append(color_names[i%ncolors])
        if shuffle:
            random.shuffle(self.colors)

        #self.network = nx.convert_node_labels_to_integers(nx.grid_graph(dim=[4, 4]), first_label = 1, ordering="sorted")
        if net == "B1":
            R = nx.cycle_graph(players)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            pos = nx.circular_layout(R)
            for i in range(1,players+1):
                if i < players-1:
                    R.add_edge(i,i+2)
                elif i == players:
                    R.add_edge(i, 2)
                else:
                    R.add_edge(i, players-i)
            self.pos = nx.circular_layout(R)
        elif net == "B2":
            R = nx.cycle_graph(players)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            pos = nx.circular_layout(R)
            for i in range(1,players+1):
                if i < players-2:
                    R.add_edge(i,i+3)
                elif i == players:
                    R.add_edge(i, 3)
                elif i == players-1:
                    R.add_edge(i, 2)
                else:
                    R.add_edge(i, players-i-1)
            self.pos = nx.circular_layout(R)
        elif net == "B3":
            R = nx.cycle_graph(players)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            pos = nx.circular_layout(R)
            for i in range(1,players+1):
                if i < players-3:
                    R.add_edge(i,i+4)
                elif i == players:
                    R.add_edge(i, 4)
                elif i == players-1:
                    R.add_edge(i, 3)
                elif i == players-2:
                    R.add_edge(i, 2)
                elif i == players-3:
                    R.add_edge(i, 1)
                else:
                    R.add_edge(i, players-i-1)
            self.pos = nx.circular_layout(R)
        elif net == "A":
            R = nx.cycle_graph(players)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            pos = nx.circular_layout(R)
            shorts1, shorts2, x = [], [], 1
            for i in range(shrt):
                s = random.randrange(3,players)
                if s not in shorts1 or s not in shorts2:
                    shorts1.append(s)
                    shorts2.append(s-2)
            x=0
            random.shuffle(shorts1)
            random.shuffle(shorts2)
            for i in range(1,players+1):
                if i not in shorts2:
                    if i < players-1:
                        R.add_edge(i,i+2)
                    elif i == players:
                        R.add_edge(i, 2)
                    else:
                        R.add_edge(i, players-i)
            R.add_edge(shorts2[0], shorts1[len(shorts2)-1])
            for i in range(len(shorts2)-1):
                R.add_edge(shorts1[i], int(shorts2[i]+(players/shrt)))
            while nx.average_degree_connectivity(R)[4]!=4:
                R = nx.cycle_graph(players)
                R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
                pos = nx.circular_layout(R)
                links = math.ceil(players*frac)
                x=0
                nlink = 0
                rew1,rew2=[],[]

                for i in range(1,players+1):
                    if i < players-1:
                        R.add_edge(i,i+2)
                    elif i == players:
                        R.add_edge(i, 2)
                    else:
                        R.add_edge(i, players-i)

                while nlink < links:
                    rewired = random.randint(1,players)
                    tmp=[]
                    for n in nx.all_neighbors(R, rewired):
                        tmp.append(n)
                    neib = random.choice(tmp)
                    rew1.append(rewired)
                    rew2.append(neib)
                    R.remove_edge(rewired, neib)
                    nlink +=1
                random.shuffle(rew2)
                random.shuffle(rew1)
                for r in range(len(rew1)):
                    if rew1[r]!=rew2[r]:
                        R.add_edge(rew1[r], rew2[r])
            self.pos = nx.circular_layout(R)

        elif net == "C1":
            d1=players-1
            tmp=[]
            links = math.ceil(players*frac)
            while d1!=0:
                if players%d1 ==0:
                    tmp.append(d1)
                d1-=1
            middle = float(len(tmp))/2
            if middle % 2 != 0:
                d1=tmp[int(middle - .5)]
                d2=int(players/d1)
            else:
                d1=tmp[int(middle-1)]
                d2=int(players/d1)
            R = nx.grid_graph(dim=[int(d1), int(d2)])
            while nx.average_degree_connectivity(R)[4]!=4:
                rew1, rew2 = [],[]
                nlink=0
                R = nx.grid_graph(dim=[int(d1), int(d2)])
                R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
                square = d1
                pos, i, row = {}, 1, 0
                for y in range(d1):
                    if y+1 <= d1/2:
                        for x in range(d2):
                            if x+1 <= d2/2:
                                pos[i] = [-x*d2+((d1-y-1)-(d2/2))**2, -y*d1+((d2-x)-(d1/2))**2]
                            elif x+1 > d2/2:
                                pos[i] = [-x*d2-((d1-y-1)-(d2/2))**2, -y*d1+(x+1-(d1/2))**2]
                            else:
                                pos[i] = [(-(d2-1)*d2)/2, y*(-(d1-1)*d1)/(d1-1)]
                            i +=1
                    if y+1 > d1/2:
                        for x in range(d2):
                            if x+1 <= d2/2:
                                pos[i] = [-x*d2+(y-(d2/2))**2, -y*d1-((d2-x)-(d1/2))**2]
                            elif x+1 > d2/2:
                                pos[i] = [-x*d2-(y-(d2/2))**2, -y*d1-(x+1-(d1/2))**2]
                            else:
                                pos[i] = [(-(d2-1)*d2)/2, y*(-(d1-1)*d1)/(d1-1)]
                            i +=1

                bot,top,left,right=[],[],[],[]
                for j in range(1, players+1):
                    if j <=d2:
                        bot.append(j)
                    if j>(players-d2):
                        top.append(j)
                    if (j-1)%(d2)==0:
                        right.append(j)
                    if j%(d2)==0:
                        left.append(j)
                for r in range(len(right)):
                    R.add_edge(left[r], right[r])
                for r in range(len(top)):
                    R.add_edge(top[r], bot[r])
                if edgeshuffle:
                    while nlink < links:
                        rewired = random.randint(1,players)
                        if R.degree(rewired) != 1:
                            tmp=[]
                            for n in nx.all_neighbors(R, rewired):
                                tmp.append(n)
                            neib = random.choice(tmp)
                            rew1.append(rewired)
                            rew2.append(neib)
                            R.remove_edge(rewired, neib)
                            nlink +=1
                    random.shuffle(rew2)
                    random.shuffle(rew1)
                    for r in range(len(rew1)):
                        if rew1[r]!=rew2[r]:
                            R.add_edge(rew1[r], rew2[r])
            self.pos = pos
            z=0
            for i in range(players):
                self.colors.append(color_names[(i+z)%ncolors])
                if (i+1)%d2 == 0:
                    z+=1
                if z == ncolors-1 and d1%3!=0:
                    z=0

        elif net == "C2":
            d1=players-1
            tmp=[]
            links = math.ceil(players*frac)
            while d1!=0:
                if players%d1 ==0:
                    tmp.append(d1)
                d1-=1
            middle = float(len(tmp))/2
            if middle % 2 != 0:
                d1=tmp[int(middle - .5)]
                d2=int(players/d1)
            else:
                d1=tmp[int(middle-1)]
                d2=int(players/d1)

            rew1, rew2 = [],[]
            nlink=0
            R = nx.grid_graph(dim=[int(d1), int(d2)])
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            square = d1
            pos, i, row = {}, 1, 0
            for y in range(d1):
                if y+1 <= d1/2:
                    for x in range(d2):
                        if x+1 <= d2/2:
                            pos[i] = [-x*d2+((d1-y-1)-(d2/2))**2, -y*d1+((d2-x)-(d1/2))**2]
                        elif x+1 > d2/2:
                            pos[i] = [-x*d2-((d1-y-1)-(d2/2))**2, -y*d1+(x+1-(d1/2))**2]
                        else:
                            pos[i] = [(-(d2-1)*d2)/2, y*(-(d1-1)*d1)/(d1-1)]
                        i +=1
                if y+1 > d1/2:
                    for x in range(d2):
                        if x+1 <= d2/2:
                            pos[i] = [-x*d2+(y-(d2/2))**2, -y*d1-((d2-x)-(d1/2))**2]
                        elif x+1 > d2/2:
                            pos[i] = [-x*d2-(y-(d2/2))**2, -y*d1-(x+1-(d1/2))**2]
                        else:
                            pos[i] = [(-(d2-1)*d2)/2, y*(-(d1-1)*d1)/(d1-1)]
                        i +=1

            bot,top,left,right=[],[],[],[]
            for j in range(1, players+1):
                if j <=d2:
                    bot.append(j)
                if j>(players-d2):
                    top.append(j)
                if (j-1)%(d2)==0:
                    right.append(j)
                if j%(d2)==0:
                    left.append(j)
            for r in range(len(right)):
                R.add_edge(left[r], right[r])
            for r in range(len(top)):
                R.add_edge(top[r], bot[r])
            if True:
                while nlink < links:
                    rewired = random.randint(1,players)
                    neib = random.randint(1,players)
                    tmp=[]
                    for n in nx.all_neighbors(R, rewired):
                        tmp.append(n)
                    if neib!=rewired and neib not in tmp:
                        rew1.append(rewired)
                        rew2.append(neib)
                        nlink +=1
                #random.shuffle(rew2)
                #random.shuffle(rew1)
                for r in range(len(rew1)):
                    if rew1[r]!=rew2[r]:
                        R.add_edge(rew1[r], rew2[r])
            self.pos = pos
            z=0
            for i in range(players):
                self.colors.append(color_names[(i+z)%ncolors])
                if (i+1)%d2 == 0:
                    z+=1
                if z == ncolors-1 and d1%3!=0:
                    z=0
        else:
            R = nx.random_regular_graph(4, players)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            self.pos = nx.spring_layout(R, scale=200, iterations=5000, k=7/players)

        self.network = R

        for i in range(players):
            self.add_new_player(self.colors[i])

    def add_new_player(self, color):
        player = Player()
        player.group = self
        player.color = color
        player.id = len(self.players)+1
        player.place = player.id
        self.players.append(player)
        if self.noise:
            if player.id not in self.noisybots:
                player.bot_person = 1
        else:
            player.bot_person = 1
        #player.player_print()

    def get_player_by_id(self, id):
        return self.players[id-1]

    def place_swap(self, req, acc):
        tmp, tmp2 = self.network.neighbors(req), self.network.neighbors(acc)
        self.network.remove_edges_from(self.network.edges(req)+self.network.edges(acc))
        self.network.add_edge(req,acc)
        lst, lst2 = [], []
        for n in tmp2:
            if n != req:
                lst.append((req, n))
        for n in tmp:
            if n != acc:
                lst2.append((acc, n))
        self.network.add_edges_from(lst+lst2)
        tmp_p= self.get_player_by_id(req).place
        #Swaps places between requesting node and accepting node
        self.get_player_by_id(req).place = self.get_player_by_id(acc).place
        self.get_player_by_id(acc).place = tmp_p
        tmp2 = self.pos[req]
        self.pos[req] = self.pos[acc]
        self.pos[acc] = tmp2
        return True

    def update_cs(self):
        for i in range(ncolors):
            clr=color_names[i]
            G=nx.Graph()
            for p in self.players:
                if clr==p.color:
                    ii = p.id
                    G.add_node(ii)
                    for n in nx.all_neighbors(self.network, ii):
                        neib = self.get_player_by_id(n)
                        if neib.color == clr:
                            jj = neib.id
                            G.add_edge(ii,jj)
            c=nx.connected_components(G)
            players = self.players
            for cc in c:
                for ccc in cc:
                    players[ccc-1].cluster_size=len(cc)
        return True

    def check_end(self):
        #Checks if the end is reached
        clrs_lst, max_clrs, color_lst =[],[],[]

        for color in color_names:
            tmp = []
            for p in self.players:
                clrs_lst.append(p.cluster_size)
                if p.color == color: tmp.append(p.cluster_size)
            color_lst.append(max(tmp))
            max_clrs.append(len(tmp))
        #colormaxs = zip(colors, color_lst)
        for p in self.players:
            p.selection = 0
            p.requested = False
            p.accepted = 0
        if gametype == "Consensus":
            if set(max_clrs).issuperset(set(clrs_lst)):
                return True
            else:
                return False
        else:
            if set(clrs_lst) == {1}:
                return True
            else:
                return False

    def progress(self):
        clrs_lst, max_clrs, color_lst =[],[],[]

        for color in color_names:
            tmp = []
            for p in self.players:
                clrs_lst.append(p.cluster_size)
                if p.color == color: tmp.append(p.cluster_size)
            color_lst.append(max(tmp))
            max_clrs.append(len(tmp))

        if gametype == "Consensus":
            avgsum = 0
            for i in range(ncolors):
                a = int(color_lst[i])
                b = int(max_clrs[i])
                avgsum += a/b
            #for p in self.players:
            return avgsum/ncolors
        else:
            for i in range(ncolors):
                G=nx.Graph()
                for p in self.players:
                    if p.cluster_size==1:
                        ii = p.id
                        G.add_node(ii)
                        for n in nx.all_neighbors(self.network, ii):
                            neib = self.get_player_by_id(n)
                            if neib.cluster_size == 1:
                                jj = neib.id
                                G.add_edge(ii,jj)

                c=nx.connected_components(G)
                tmp = []
                for cc in c:
                    tmp.append(len(cc))
                if len(tmp) != 0:
                    correctlocals = max(tmp)
                    return correctlocals/players
                return 0

    def print_graph(self, name, prog):
        #for i in self.players:
        #    print(i.id, " : ", i.place, " : ", i.color, " : ", i.cluster_size)
        print(nx.average_shortest_path_length(self.network))
        G=nx.Graph()
        clrs_lst = []
        for color in color_names:
            tmp = []
            for p in self.players:
                clrs_lst.append(p.cluster_size)
                if p.color == color: tmp.append(p.cluster_size)
            for p in self.players:
                if color==p.color and p.cluster_size == max(tmp):
                    ii = p.id
                    G.add_node(ii)
                    for n in nx.all_neighbors(self.network, ii):
                        neib = self.get_player_by_id(n)
                        if neib.color == color:
                            jj = neib.id
                            G.add_edge(ii,jj)
        c=list(nx.connected_components(G))
        pos = {}
        z=0
        plt.figure("Largest clusters: "+str(name)+ " "+str(prog)+ " "+str(frac))
        for color in range(ncolors):
            nx.draw_networkx_nodes(self.network, self.pos, nodelist=c[z], node_color=drawcolors[color])
            z+=1
        nx.draw_networkx_edges(self.network,self.pos,width=1.0,alpha=0.5)
        #nx.draw_networkx_labels(self.network,self.pos,width=1.0,alpha=0.5)
        '''z=1
        plt.figure("End state: "+str(name)+ " "+str(prog))
        for color in self.colors:
            nx.draw_networkx_nodes(self.network, self.pos, nodelist=[z], node_color=drawcolors[color-1])
            z+=1
        nx.draw_networkx_edges(self.network,self.pos,width=1.0,alpha=0.5)
        nx.draw_networkx_labels(self.network,self.pos,width=1.0,alpha=0.5)'''

        #plt.show()
        #time.sleep(2)


#Gameloop

def game(ai, games, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle):
    '''
    function game gathers the results from each game
    param: ai (Boolean, makes bots either greedy or noisy)
    param: games (Int, number of games to be played)
    param: shuffle (Boolean, shuffles the initial colors)
    param: info_vis, info_clust (Boolean, allows the bots information of second neighbor(vis) or their local cluster sizes (clust))
    param: noisyness (Int, the percentage of noisy desicions made by bots)
    param: threshold (Float, the threshold value for critical cluster size)
    param: net (Str, decides which topology will be used)
    param: edgeshuffle (Boolean, used in torus with random boundary)

    rdys = list, Number of games that exceed 60% progress
    progs = list, prodress of the game in the end
    plott = 2dim list, progress each round
    runs = list, the round when game was solved
    dist = list, the average distances in each network

    Returns: list[runs, rdys, progs, plott, dist]
    '''
    rdys = []
    progs = []
    plott = []
    runs = []
    dist = []
    for x in range(0,games):
        tmp = game_loops(x,ai, shuffle, info_vis, info_clust, rdys, progs, plott, noisyness, threshold, net, edgeshuffle)
        runs.append(tmp[0])
        rdys = tmp[1]
        progs = tmp[2]
        plott = tmp[3]
        dist.append(tmp[4])
    return [runs,rdys,progs,plott, dist]

def game_loops(x, noise, shuffle, info_vis, info_clust, rdys, progs, plott, noisyness, threshold, net, edgeshuffle):
    '''
    function game_loops runs a single game and stores the necessary data

    param: ai (Boolean, makes bots either greedy or noisy)
    param: games (Int, number of games to be played)
    param: shuffle (Boolean, shuffles the initial colors)
    param: info_vis, info_clust (Boolean, allows the bots information of second neighbor(vis) or their local cluster sizes (clust))
    param: noisyness (Int, the percentage of noisy desicions made by bots)
    param: threshold (Float, the threshold value for critical cluster size)
    param: net (Str, decides which topology will be used)
    param: edgeshuffle (Boolean, used in torus with random boundary)

    rdys = list, Number of games that exceed 60% progress
    progs = list, prodress of the game in the end
    plott = 2dim list, progress each round
    runs = list, the round when game was solved
    dist = list, the average distances in each network

    Returns: list[time, rdys, progs, plott, nx.average_shortest_path_length(graf.network)]
    '''
    graf = Group(players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle)
    graf.update_cs()
    #graf.print_graph(x, graf.progress())
    plott.append([graf.progress()])
    #print(graf.progress())
    time = rounds
    for i in range(1,rounds+1):
        if not graf.check_end():
            #graf.threshold = threshold/(players/rounds)
            color_in_turn = color_names[i % ncolors]
            for p in graf.players:
                if p.color == color_in_turn:
                    p.request()
            for p in graf.players:
                if p.requested:
                    p.accept()
            for p in graf.players:
                if p.accepted != 0:
                    graf.place_swap(p.id, p.accepted)
            graf.update_cs()

        '''else:
            if time == rounds:
                time = i
                rdys.append(i)'''
        plott[x].append(graf.progress())
    #if graf.progress() > 0.98:
    #graf.print_graph(x, graf.progress())
    #graf.print_graph(x, graf.progress())
    #print(nx.average_degree_connectivity(graf.network))
    #print(graf.progress())
    progs.append(graf.progress())
    if graf.progress() > 0.6:
        rdys.append(1)
    return [time, rdys, progs, plott, nx.average_shortest_path_length(graf.network)]

'''
The rest of this script is different runs and their PLOTTING
the functions are named with different setups that were tested

Basic use: Run function clust_param with needed parameters

Function plotparam plots the average end progress over a variable parameter
'''


def noise_onoff(n_games):
    games = n_games
    shuffle = True
    info_vis = True
    info_clust = True
    game1 = game(False, games,shuffle,info_vis, info_clust, 0)
    game2 = game(True, games,shuffle,info_vis, info_clust, 100)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    runs1 = game2[0]
    rdys1 = game2[1]
    progs1 = game2[2]
    plott1 = game2[3]
    if len(rdys)==0:
        rdys.append(0)
    if len(rdys1)==0:
        rdys1.append(0)
    print(" RESULTS (BOT AI NOISE off/on, Random:", shuffle,", Info (Vision/CS): ", info_vis, "/", info_clust,")")
    print("AVG ROUNDS: ", np.average(runs), "/", np.average(runs1))
    print("COMPLETIONS: ", progs.count(1.0)/games, "/", progs1.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs), "/", np.average(progs1))
    print("VAR PROGRESS: ", np.var(progs), "/", np.var(progs1))
    print("MAX PROGRESS: ", max(progs), "/", max(progs1))
    print("MIN PROGRESS: ", min(progs), "/", min(progs1))
    print("GOALTIME ", np.average(rdys), "/", np.average(rdys1))
    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    #plt.figure("Noise on/off")
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "'AI NOISE OFF'")
    lst1 = np.asarray(plott1).T.tolist()
    avgs1 = []
    for j in range(0, len(lst1)):
        avgs1.append(np.average(lst1[j]))
    plt.plot(np.arange(1,len(avgs1)+1), avgs1, label = "'AI NOISE ON'")
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')

def noise_amount(n_games):
    games = n_games
    shuffle = True
    info_vis = True
    info_clust = True
    game1 = game(True, games,shuffle,info_vis, info_clust, 10)
    game2 = game(True, games,shuffle,info_vis, info_clust, 30)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    runs1 = game2[0]
    rdys1 = game2[1]
    progs1 = game2[2]
    plott1 = game2[3]
    if len(rdys)==0:
        rdys.append(0)
    if len(rdys1)==0:
        rdys1.append(0)
    print(" RESULTS (BOT AI NOISE 10/30, Random:", shuffle,", Info (Vision/CS): ", info_vis, "/", info_clust,")")
    print("AVG ROUNDS: ", np.average(runs), "/", np.average(runs1))
    print("COMPLETIONS: ", progs.count(1.0)/games, "/", progs1.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs), "/", np.average(progs1))
    print("VAR PROGRESS: ", np.var(progs), "/", np.var(progs1))
    print("MAX PROGRESS: ", max(progs), "/", max(progs1))
    print("MIN PROGRESS: ", min(progs), "/", min(progs1))
    print("GOALTIME ", np.average(rdys), "/", np.average(rdys1))
    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "'AI NOISE 10%'")
    lst1 = np.asarray(plott1).T.tolist()
    avgs1 = []
    for j in range(0, len(lst1)):
        avgs1.append(np.average(lst1[j]))
    plt.plot(np.arange(1,len(avgs1)+1), avgs1, label = "'AI NOISE 30%'")
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')

def vision_onoff(n_games, t1):
    games = n_games
    shuffle = True
    info_vis = True
    noise = False
    info_clust = True
    game1 = game(noise, games,shuffle,True, True, 0, t1)
    game2 = game(noise, games,shuffle,False, True, 0, t1)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    runs1 = game2[0]
    rdys1 = game2[1]
    progs1 = game2[2]
    plott1 = game2[3]
    if len(rdys)==0:
        rdys.append(0)
    if len(rdys1)==0:
        rdys1.append(0)
    print(" RESULTS (BOT AI VISION on/off, Random:", shuffle,", Noise: ", noise, ")")
    print("AVG ROUNDS: ", np.average(runs), "/", np.average(runs1))
    print("COMPLETIONS: ", progs.count(1.0)/games, "/", progs1.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs), "/", np.average(progs1))
    print("VAR PROGRESS: ", np.var(progs), "/", np.var(progs1))
    print("MAX PROGRESS: ", max(progs), "/", max(progs1))
    print("MIN PROGRESS: ", min(progs), "/", min(progs1))
    print("GOALTIME ", np.average(rdys), "/", np.average(rdys1))
    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "'AI VIS ON, CLUST ON' "+str(t1))
    lst1 = np.asarray(plott1).T.tolist()
    avgs1 = []
    for j in range(0, len(lst1)):
        avgs1.append(np.average(lst1[j]))
    plt.plot(np.arange(1,len(avgs1)+1), avgs1, label = "'AI VIS OFF, CLUST ON '"+str(t1))
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')

def clust_onoff(n_games):
    games = n_games
    shuffle = False
    info_vis = True
    noise = False
    info_clust = True
    game1 = game(noise, games,shuffle,True, True, 0, 0.5)
    game2 = game(noise, games,shuffle,False, False, 0, 0.5)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    runs1 = game2[0]
    rdys1 = game2[1]
    progs1 = game2[2]
    plott1 = game2[3]
    if len(rdys)==0:
        rdys.append(0)
    if len(rdys1)==0:
        rdys1.append(0)
    print(" RESULTS (BOT AI VISION on/off, Random:", shuffle,", Noise: ", noise, ")")
    print("AVG ROUNDS: ", np.average(runs), "/", np.average(runs1))
    print("COMPLETIONS: ", progs.count(1.0)/games, "/", progs1.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs), "/", np.average(progs1))
    print("VAR PROGRESS: ", np.var(progs), "/", np.var(progs1))
    print("MAX PROGRESS: ", max(progs), "/", max(progs1))
    print("MIN PROGRESS: ", min(progs), "/", min(progs1))
    print("GOALTIME ", np.average(rdys), "/", np.average(rdys1))
    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    #plt.figure("Cluster information on/off")
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "'AI CLUST ON, VIS ON'")
    lst1 = np.asarray(plott1).T.tolist()
    avgs1 = []
    for j in range(0, len(lst1)):
        avgs1.append(np.average(lst1[j]))
    plt.plot(np.arange(1,len(avgs1)+1), avgs1, label = "'AI INFO OFF'")
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')

def clust_param(n_games, t1, net, edgeshuffle, name, clust, noise, noisyness):
    games = n_games
    shuffle = False
    info_vis = False
    noise = noise
    info_clust = clust
    game1 = game(noise, games,shuffle,info_vis, info_clust, noisyness, t1, net, edgeshuffle)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    #print(game1)
    if len(rdys)==0:
        rdys.append(0)
    print(" RESULTS (THRESHOLD", t1, "Random:", shuffle,", Noise: ", noise, ")")
    print("AVG DIST: ", np.average(game1[4]))
    print("AVG ROUNDS: ", np.average(runs))
    print("COMPLETIONS: ", progs.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs))
    print("VAR PROGRESS: ", np.var(progs))
    print("MAX PROGRESS: ", max(progs))
    print("MIN PROGRESS: ", min(progs))
    #print("GOALTIME ", np.average(rdys))
    print("60% Progress in ", len(rdys))


    paramlog.append([noisyness, np.average(progs)])
    #paramlog.append([t1, np.average(progs)])

    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    plt.figure("Progress")
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = str(noisyness) + "% noise, thres "+str(t1)+name)
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')
    return([noisyness, np.average(progs)])


#plt.figure("Progress")

paramlog = []

'''
NETWORKS:

A = Circular small world
B1 = Regular ring with shortcuts of 2 edges
B2 = Regular ring with shortcuts of 3 edges
B3 = Regular ring with shortcuts of 4 edges
C1 = Torus with rewired small world edges
C2 = Torus with added small world edges
D = Random generated graph
'''
#clust_param(50, 1, "C", " mesh noise: ", False, True, 100)
#clust_param(2, 0.4, "D", False, " random ", True, False, 15)
#clust_param(10, 0.4, "A", False, " ring sw ", True, False, 15)
#clust_param(100, 0.4, "C", True, " mesh sw", True, False, 5)
clust_param(500, 0.5, "C2", True, " torus REW "+str(frac), True, False, 5)
clust_param(500, 0.5, "C2", True, " torus REW "+str(frac), True, False, 15)
clust_param(500, 0.5, "C2", True, " torus REW "+str(frac), True, False, 25)
clust_param(500, 0.5, "C2", True, " torus REW "+str(frac), True, False, 30)



#clust_param(10, 0.5, "C2", True, " mesh sw ADD", True, False, 5)

#clust_param(100, 0.5, "C", True, " mesh sw randomAI", True, True, 100)
#clust_param(10, 0.4, "C", False, " mesh torus ", True, False, 15)
#clust_param(1000, 0.4, "C", True, " mesh sw", True, True, 100)
#clust_param(50, 0.6, "B1", False, " 3ring ", True, False, 15)
#clust_param(50, 0.6, "B2", False, " 4ring ", True, False, 15)
#clust_param(100, 0.4, "B3", False, " 5ring ", True, False, 15)


def plotparam():
    params = np.arange(0,30,5)
    params2 = np.arange(0.3,0.6,0.1)
    for g in params2:
        paramlog = []
        for t in params:
            #clust_param(50, 1, "A", " noise: "+str(t), False, True, t)
            paramlog.append(clust_param(25, g, "C", True, " mesh sw", True, False, t))
        plt.figure("Progress with noises")
        lst = np.asarray(paramlog).T.tolist()
        plt.plot(lst[0], lst[1], label = "Threshold progress "+str(g))
    plt.xlabel('Noise%')
    plt.ylabel('Average progress')
    plt.axis((0,50,0,1))
    #for xy in paramlog:
    #    plt.annotate('(%s, %s)' % (round(xy[0],2), round(xy[1],2)), xy=xy, textcoords='data')
    plt.legend(loc='best')

#plotparam()


plt.ion()
plt.show()
input()
#CSV writing/parsing etc
