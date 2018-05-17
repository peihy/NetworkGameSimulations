from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import networkx as nx
from operator import itemgetter
import time as timer
import math

'''
Initial setup

rounds: Number of rounds for each game
players: Network size (number of agents)
frac: Fraction of rewired edges n_edges = players*frac
ncolors: number of colors
color_names: "names" of the colors
drawcolors: colorcodes for plotting purposes
colors: playercolors listed
Rule: The adhoc rule that inhibits interactions between clusters of large size (>=60% of maximum clustersize)

'''
rounds = 21
players = 30
Rule = True
frac = 0.1
ncolors = 3
color_names = [1, 2, 3]
drawcolors = ['r', 'g', 'b']
colors=[]

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
        '''
        For clarity, get the parameters as their real names
        '''
        lam =self.group.params[5][0] #lambda_a
        delta=self.group.params[5][3] #delta_a
        beta = self.group.params[5][2] #beta_a
        alpha= self.group.params[5][1] #alpha_a
        requestors = [] #list for neighbors that request the current ego
        for n in nx.all_neighbors(self.group.network, self.id):
            neib = self.group.get_player_by_id(n)
            nc=0
            oc=0
            csum=0
            if neib.selection == self.id:
                for nn in nx.all_neighbors(self.group.network, self.id):
                    neib2 = self.group.get_player_by_id(nn)
                    if neib2.color == neib.color:
                        nc+=1
                        csum+=neib2.cluster_size
                '''
                Naming the variables as they are in formula on p.9 on the manuscript, for clarity
                '''
                si = self.cluster_size/(players/3) #The scaled clystersize of the ego. ego's cs divided by the maximum possible cs
                sj = neib.cluster_size/(players/3) #The scaled clustersize of the alter. alter's cs divided byt the maximum possible cs
                nicj = (csum/nc)/(players/3) #The average clustersize of the alter's color's agents in the local neighborhood
                wrcom = np.exp(lam+(nicj*delta)+sj*beta+si*alpha) # the exponential for the probability matching model
                #Add the neighbour id and the corresponding exponential to the list for the final formulation
                requestors.append([neib.id, wrcom])



        if len(requestors) > 0:

            #if wr>ws or random.randrange(100)<(self.group.noisyness):
            dsum,nsum=1,0
            for n in requestors:
                dsum+=n[1]
            chp=0

            #PICK RANDOM IF ANY
            '''
            As discovered from data analysis of the experiment, there is no preference in choosing the requester when multiple request are present.
            '''
            r2 = np.random.rand()
            if (dsum-1)/dsum>=r2:
                self.accepted = random.choice(requestors)[0]
                for rb in requestors:
                    if rb[1]>chp:
                        chp=rb[1]
                        self.accepted=rb[0]

        self.requested = False

    def request(self):

        '''
        THE REQUEST
        '''
        requestables = []
        wr=0
        cs = 100000
        for n in nx.all_neighbors(self.group.network, self.id):
            neib = self.group.get_player_by_id(n)
            '''
            For clarity, get the parameters as their real names as in formula in p.9 of manuscript
            '''
            lam =self.group.params[0] #lambda_r
            delta=self.group.params[3] #delta_r
            beta = self.group.params[2] #beta_r
            alpha= self.group.params[1] #alpha_r
            nc=0
            csum=0
            mac=neib.cluster_size
            if neib.color != self.color:
                for nn in nx.all_neighbors(self.group.network, self.id):
                    neib2 = self.group.get_player_by_id(nn)
                    if neib2.color == neib.color:
                        nc+=1
                        csum+=neib2.cluster_size

                '''
                Naming the variables as they are in formula on p.9 on the manuscript, for clarity
                '''
                si = self.cluster_size/(players/3) #The scaled clystersize of the ego. ego's cs divided by the maximum possible cs
                sj = neib.cluster_size/(players/3) #The scaled clustersize of the alter. alter's cs divided byt the maximum possible cs
                nicj = (csum/nc)/(players/3) #The average clustersize of the alter's color's agents in the local neighborhood
                wrcom = np.exp(lam+(nicj*delta)+(sj*beta)+(si*alpha))
                '''
                The ad-hoc rule that inhibits the exchanges between agents with large clustersize (by making the probability as 0)
                Used then the global boolean value "Rule" is True
                '''
                if neib.cluster_size/(players/3) > 0.6 and self.cluster_size/(players/3) > 0.6 and Rule:
                    wrcom=0
                    #Not adding the neighbour to the list of possible requestables
                else:
                    requestables.append([neib.id, wrcom])

        if len(requestables) != 0:
            dsum,nsum=1,0
            for n in requestables:
                dsum+=n[1]
            #print((dsum-1)/(dsum))
            r2 = np.random.rand()
            random.shuffle(requestables)
            chp=0
            for n in requestables:
                if (chp+n[1])/((dsum))>=r2:
                    self.selection = n[0]
                    break
                else:
                    chp+=n[1]

            '''
            SOME BUG WITH ID 30 FIX
            '''
            if self.selection != 0:
                self.group.get_player_by_id(self.selection).requested = True

    def player_print(self):
        print(self.id, self.color, self.place, self.bot_person)

class Group(object):
    '''
    The group object that controls the flow of the game. Basically all the gamefunctions
    '''
    def __init__(self, players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle, params):
        self.mapper = []
        self.players = []
        self.noise = noise
        self.noisyness = noisyness
        self.info_vis = info_vis
        self.info_clust = info_clust
        self.threshold = threshold
        self.noisybots = []
        self.params= params
        color_names = [1, 2, 3]

        self.colors=[]
        if net != "C1" or net != "C2":
            for i in range(players):
                self.colors.append(color_names[i%ncolors])
        if shuffle:
            random.shuffle(self.colors)

        '''
        Setting the self network according to the given parameter
        '''
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
            posC = {}
            k = 1
            rr = 20
            const = 1
            shift= 1/(d1-1)
            ww,xx,yy = 1,0,0
            for i in range(d1):
                if i == 0 or i == d1-1:
                    const = 2
                else:
                    const = 3+shift*(((2)-(i))**2)/(d1**2)
                ww +=2
                for pp in range(1,d2+1):
                    '''if i == 0 or i == d1-1:
                        xx,yy=1,1
                    else:
                        if pp == 1 or pp == d2/2+1:
                            yy = 20
                            xx = 20
                        else:
                            xx = -5
                            yy = 20'''
                    posC[k] = [(math.cos(((const)*pp -1) * (2*math.pi/(d2*(const)))  - math.pi/2.0)* rr)*ww+xx, (math.sin((pp*(const) -1) * (2*math.pi/(d2*(const))) - math.pi/2.0) * rr)*ww+yy]
                    k+=1
            pos = posC
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
                    if neib!=rewired and neib not in tmp and neib not in rew1 and neib not in rew2 and rewired not in rew1 and rewired not in rew2:
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
            self.colors=[]

            for i in range(players):
                self.colors.append(color_names[(i+z)%ncolors])
                if i+1 >= players-d2 and d1%2==1:
                    color_names = [2, 1, 3]
                    z+=1
                else:
                    if (i+1)%d2 == 0:
                        z+=1
                    if z == ncolors-1 and d1%3!=0:
                        z=0
        else:
            #R = nx.random_regular_graph(4, players)
            R = nx.connected_watts_strogatz_graph(players,4,0.4)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            self.pos = nx.spring_layout(R, scale=200, iterations=5000, k=7/players)

        self.network = R
        #print(self.colors)
        if not (self.colors.count(1)==self.colors.count(2) and self.colors.count(2)==self.colors.count(3)):
            self.colors=[]
            for i in range(players):
                self.colors.append(color_names[(i)%ncolors])
        for i in range(players):
            self.add_new_player(self.colors[i])

    def add_new_player(self, color):
        player = Player()
        player.group = self
        player.color = color
        player.id = len(self.players)+1
        player.place = player.id
        self.players.append(player)
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
        for p in self.players:
            p.selection = 0
            p.requested = False
            p.accepted = 0
        if set(max_clrs).issuperset(set(clrs_lst)):
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

        avgsum = 0
        for i in range(ncolors):
            a = int(color_lst[i])
            b = int(max_clrs[i])
            avgsum += a/b
        return avgsum/ncolors

    def print_graph(self, name, prog):
        G=nx.Graph()
        clrs_lst = {}
        for color in color_names:
            tmp = []
            for p in self.players:
                clrs_lst[p.id]=p.cluster_size
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
        #z=0
        z=1
        for color in self.colors:
            nx.draw_networkx_nodes(self.network, self.pos, nodelist=[z], node_color=drawcolors[color-1])
            if z<players:
                z+=1
        nx.draw_networkx_edges(self.network,self.pos,width=1.0,alpha=0.5)
        nx.draw_networkx_labels(self.network,self.pos,labels=clrs_lst,width=1.0,alpha=0.5)
        plt.title(str(drawcolors[name-1])+str(prog))
        plt.ion()
        plt.show()

#Gameloop

def game(ai, games, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle, params, animate):
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

    Returns: runs, rdys, progs, plott, dist
    '''
    rdys = []
    progs = []
    acts=[]
    rs=[]
    plott = []
    runs = []
    dist = []
    accs=[[] for i in range(int(players/3))]
    reqs=[[] for i in range(int(players/3))]
    for x in range(0,games):
        lastrun, rdys, progs, plott, lastdist, lastacts, lastrs, accs, reqs = game_loops(x,ai, shuffle, info_vis, info_clust, rdys, progs, plott, acts, noisyness, threshold, net, edgeshuffle, params,accs,reqs, animate)
        runs.append(lastrun)
        dist.append(lastdist)
        acts.append(lastacts)
        rs.append(lastrs)
    return runs,rdys,progs,plott, dist, acts, rs, accs,reqs

def game_loops(x, noise, shuffle, info_vis, info_clust, rdys, progs, plott,acts, noisyness, threshold, net, edgeshuffle, params, accs,reqs, animate):
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
    graf = Group(players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle, params)
    graf.update_cs()
    plott.append([graf.progress()])
    g_act=[]
    r_act=[]
    time = rounds

    for i in range(1,rounds+1):
        tmp3=[]
        tmp2=[]
        if not graf.check_end():
            if animate:
                plt.clf()
                graf.print_graph(1, 1)
                plt.pause(0.05)
            color_in_turn = color_names[i % ncolors]
            for p in graf.players:
                if p.color == color_in_turn:
                    p.request()
                    if p.selection !=0:
                        reqs[p.cluster_size-1].append(1)
                        tmp2.append(1)
                    else:
                        reqs[p.cluster_size-1].append(0)
                        tmp2.append(0)
            for p in graf.players:
                if p.requested:
                    p.accept()
                    if p.accepted !=0:
                        accs[p.cluster_size-1].append(1)
                        tmp3.append(1)
                    else:
                        #print(p.id)
                        accs[p.cluster_size-1].append(0)
                        tmp3.append(0)
                    p.requested=False
            for p in graf.players:
                if p.accepted != 0:
                    graf.place_swap(p.id, p.accepted)
            graf.update_cs()

        if len(tmp3)!=0:
            g_act.append(sum(tmp3)/len(tmp3))
        else:
            if not graf.check_end():
                g_act.append(0)
            else:g_act.append(np.nan)
        if len(tmp2)!=0:
            r_act.append(sum(tmp2)/len(tmp2))
        else:
            if not graf.check_end():
                r_act.append(0)
            else:r_act.append(np.nan)
        if not sum(tmp2)>=len(tmp3): print(sum(tmp2)>=len(tmp3), sum(tmp2), len(tmp3), tmp2, tmp3)
        graf.check_end()
        plott[x].append(graf.progress())

    progs.append(graf.progress())
    if graf.progress() > 0.6:
        rdys.append(1)
    return [time, rdys, progs, plott, nx.average_shortest_path_length(graf.network), g_act, r_act, accs, reqs]

'''
The rest of this script is different runs and their PLOTTING
the functions are named with different setups that were tested

Basic use: Run function clust_param with needed parameters

Function plotparam plots the average end progress over a variable parameter
'''

def clust_param(games, t1, net, edgeshuffle, name, clust, noise, noisyness, params, animate):

    runs, rdys, progs, plott, dist, acts, rs, accs, reqs = game(noise, games,False,False, False, noisyness, t1, net, edgeshuffle, params, animate)

    '''
    Printing the information of current run of clust_param
    '''
    if len(rdys)==0:
        rdys.append(0)
    print(" RESULTS (THRESHOLD", name)
    print("AVG DIST: ", np.average(dist))
    print("AVG ROUNDS: ", np.average(runs))
    print("COMPLETIONS: ", progs.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs))
    print("VAR PROGRESS: ", np.var(progs))
    print("MAX PROGRESS: ", max(progs))
    print("MIN PROGRESS: ", min(progs))
    print("60% Progress in ", len(rdys))


    #PLOTTING
    '''
    Making the plots for the ACP, Acceptance- and Request activity and plotting the distribution of ACP at the end of given rounds
    '''
    barcol='#356bc4' #Nice hex color code for uniform style

    lst = np.asarray(plott).T.tolist()
    fig=plt.figure("Progress")
    avgs = []
    yerr = []
    for j in range(0, len(lst)-1):
        avgs.append(np.average(lst[j]))
        yerr.append(np.std(lst[j])/np.sqrt(len(lst[j])))

    plt.title('Simulation: 30 players')
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = name,color=barcol)
    plt.errorbar(np.arange(1,len(avgs)+1), avgs, yerr=yerr, fmt='o', ecolor='r')
    plt.xticks(np.arange(1,len(avgs)+1), np.arange(1,len(avgs)+1))
    plt.xlabel('Round n')
    plt.ylabel('Activity')
    lst=[]
    lst1 = np.asarray(list(acts)).T.tolist()
    avgs = []
    for j in range(0, len(lst1)):
        avgs.append(np.nanmean(lst1[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "Accepting Activity ", color='grey', linestyle='--')
    lst1 = np.asarray(list(rs)).T.tolist()
    avgs = []
    for j in range(0, len(lst1)):
        avgs.append(np.nanmean(lst1[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "Requesting Activity ", color='g', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Activity')
    plt.legend(loc="lower left",prop={'size': 8})
    plt.grid()
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig('10simulation_30p_rule.eps')
    '''
    Activity for requesting as function of clustersize (histogram)
    '''
    xaxs=np.linspace(1,players/3,players/3)
    plt.figure("Activity: Request")
    plt.title("Activity: Request")
    plt.xlabel('S_i')
    plt.ylabel('Activity')
    avgs=[]
    for i in reqs:
        avgs.append(np.average(i))
    plt.grid()
    plt.bar(xaxs,avgs,color=barcol)
    '''
    Activity for accepting as function of clustersize (histogram)
    '''
    plt.figure("Activity: Accept")
    plt.title("Activity: Accept")
    plt.xlabel('S_i')
    plt.ylabel('Activity')
    plt.grid()
    avgs=[]
    for i in accs:
        avgs.append(np.average(i))
    plt.bar(xaxs,avgs,color=barcol)

    '''
    ACP ditribution
    '''
    plt.figure("Progress: Distribution")
    plt.title("Prog dist")
    plt.xlabel('prog')
    plt.ylabel('%')
    plt.grid()
    plt.hist(progs, bins=10, normed=True,color=barcol)
    return([noisyness, np.average(progs),progs.count(1.0)/games])



'''
NETWORKS:

A = Circular small world
B1 = Regular ring with shortcuts of 2 edges
B2 = Regular ring with shortcuts of 3 edges
B3 = Regular ring with shortcuts of 4 edges
C1 = Torus with rewired small world edges
C2 = Torus with added small world edges
D = Random generated graph with regular degree 4
'''

'''
Format for the parameters used in the model (p.9)
Request: P_gamma/P_0 = exp{lambda_r+alpha_r*s_i+beta_r*s_j+delta_r*<s_i(c_j)>}
Accepting: P_gamma/P_0 = exp{lambda_a+alpha_a*s_i+beta_a*s_j+delta_a*<s_i(c_j)>}

parameters = [lambda_r,alpha_r,beta_r,delta_r,0,[lambda_a,alpha_a,beta_a,delta_a,0]]
Zeros are legacy from another model attribute
'''

prms=[ 2.9635355,  -4.68414371, -4.52773696,  3.85244056, 0, [ 2.70687544, -4.15585307, -4.39269945,  4.76448243, 0]]

'''
clust_param
param 1: number of games
param 2: Threshold for adhoc rule (Legacy and redundant)
param 3: Network type as string
param 4: Edgeshuffle boolean (Legacy and redundant)
param 5: Title for plots i.e. when making different games and plotting on the same graph
param 6: Clust boolean value, Whether the agents know the information of local neighbourhood or not (Legacy and redundant)
param 7: Noise boolean, whether there are agents with random policy (Legacy and redundant)
param 8: Noisyness as percents, used when previous parameter is True (Legacy and redundant)
param 9: The parameters for the model as described previously in the required format.
param 10: Whether the game will be viasualized as an animation
'''
run=clust_param(250, 0.6, "C2", False, "Experimental parameters ", False, False, 100, prms, False)

'''
Finally make the plots visible for evaluation
'''
plt.ion()
plt.show()
input()
