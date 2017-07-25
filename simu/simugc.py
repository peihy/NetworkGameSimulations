import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import networkx as nx
from operator import itemgetter

'''
Initial setup
'''
rounds = 30
players = 21
ncolors = 3
color_names = [1, 2, 3, 4, 5, 6, 7]
drawcolors = ['r', 'g', 'b', 'y', 'black', 'purple']

colors=[]
b=1
for i in range(players):
    if i+1 > b*players/ncolors:
        b+=1
    colors.append(color_names[b-1])
#random.shuffle(colors)

B = nx.cycle_graph(players)
B = nx.convert_node_labels_to_integers(B, first_label = 1, ordering="sorted")
a_pos = nx.circular_layout(B)
for i in range(1,players+1):
    if i < players-1:
        B.add_edge(i,i+2)
    elif i == players:
        B.add_edge(i, 2)
    else:
        B.add_edge(i, players-i)
plt.figure("Initial")
z=1
for color in colors:
    nx.draw_networkx_nodes(B, a_pos, nodelist=[z], node_color=drawcolors[color-1])
    z+=1
nx.draw_networkx_edges(B,a_pos,width=1.0,alpha=0.5)

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

    #Handle the different phases for each players
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
            #if self.in_largest():
            #    return
            if choice <= self.group.noisyness:
                for n in nx.all_neighbors(self.group.network, self.id):
                    neib = self.group.get_player_by_id(n)
                    if neib.color != self.color:
                        requestables.append(neib)
                if len(requestables) > 0:
                    self.selection = random.choice(requestables).id
                    self.group.get_player_by_id(self.selection).requested = True


    def greedy(self, string):
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
                    #self.accepted = random.choice(requestors)
                    pos = []
                    for i in requestors:
                        #if self.common_neighbour(i):
                        #    self.accepted = i
                        if i in smart:
                            pos.append(i)
                    if len(pos)>0: self.accepted = random.choice(pos)
                    if random.randrange(100) < 10:
                        self.accepted = random.choice(requestors)
            self.requested = False
        else:
            requestables = []
            if self.in_largest() and self.group.info_clust:
                return
            for n in nx.all_neighbors(self.group.network, self.id):
                neib = self.group.get_player_by_id(n)
                if neib.color != self.color:
                    requestables.append(neib)
            if smart != 0:
                self.selection = random.choice(smart)
                self.group.get_player_by_id(self.selection).requested = True
            '''else:
                if len(requestables) > 0:
                    self.selection = random.choice(requestables).id
                    self.group.get_player_by_id(self.selection).requested = True'''


    def common_neighbour(self, req):
        #IS THIS ACCEPTABLE
        return False
        chk = True
        reqcol = self.group.get_player_by_id(req)
        for n in nx.all_neighbors(self.group.network, self.id):
            neib = self.group.get_player_by_id(n)
            if neib.color == reqcol and neib.id != req:
                chk = False
        for n2 in nx.all_neighbors(self.group.network, req):
            neibneib = self.group.get_player_by_id(n2)
            #if neib.id == neibneib.id:
            if neibneib.color == self.color and neibneib.id != self.id:
                chk = False
        return chk

    def smart_req(self):
        choice = []
        second = []
        for n in nx.all_neighbors(self.group.network, self.id):
            tmp=[]
            neib = self.group.get_player_by_id(n)
            if neib.color != self.color:
                for n2 in nx.all_neighbors(self.group.network, neib.id):
                    neibneib = self.group.get_player_by_id(n2)
                    if self.id != neibneib.id and neibneib.color == self.color:
                        tmp.append(neibneib.id)
                if len(tmp) < self.cluster_size:
                    choice.append(neib.id)
                if len(tmp) <= self.cluster_size:
                    second.append(neib.id)

        if len(choice)>0:return choice
        return [0]


    def in_largest(self):
        G=nx.Graph()
        if self.cluster_size==1:
            ii = self.id
            G.add_node(ii)
            for n in nx.all_neighbors(self.group.network, ii):
                neib = self.group.get_player_by_id(n)
                if neib.cluster_size == 1:
                    jj = neib.id
                    G.add_edge(ii,jj)

            c=nx.connected_components(G)
            cnt=0
            for i in c:
                cnt=len(i)
            prob = cnt/players
            #if prob < self.group.progress():
            #    return False
            if prob > self.group.threshold:
                return True
        #if random.randrange(100) <= (prob**2)*100:
        #    return True
        return False

    def player_print(self):
        print(self.id, self.color, self.place, self.bot_person)

class Group(object):
    def __init__(self, players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold):
        self.mapper = []
        self.players = []
        self.noise = noise
        self.noisyness = noisyness
        self.info_vis = info_vis
        self.info_clust = info_clust
        self.threshold = threshold
        self.noisybots = []
        for x in range(3):
            self.noisybots.append(random.randint(1,players))
        '''
        SHUFFLE
        '''
        colors=[]
        b=1
        for i in range(players):
            if i+1 > b*players/ncolors:
                b+=1
            colors.append(color_names[b-1])
        if shuffle:
            random.shuffle(colors)

        #self.network = nx.convert_node_labels_to_integers(nx.grid_graph(dim=[4, 4]), first_label = 1, ordering="sorted")
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
        self.network = R
        for i in range(players):
            self.add_new_player(colors[i])

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
        #print("Swapping places:", req, acc)
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

        for color in colors:
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

        if set(clrs_lst) == {1}:
            return True
        else:
            return False

    def progress(self):
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


    def print_graph(self):
        for i in self.players:
            print(i.id, " : ", i.color, " : ", i.cluster_size)
        plt.clf()
        pos = {}
        row, i = 0, 1
        square = 8
        for node in self.network.nodes():
            if node+1 > (row+1)*square:
                row +=1
            pos[i] = [(node-(row*square))*200/square, row*200/square]
            i +=1
        plt.ion()
        plt.figure("Graph")
        nx.draw(self.network, with_labels=True)



#Gameloop

def game(ai, games, shuffle, info_vis, info_clust, noisyness, threshold):
    rdys = []
    progs = []
    plott = []
    runs = []
    for x in range(0,games):
        tmp = game_loops(x,ai, shuffle, info_vis, info_clust, rdys, progs, plott, noisyness, threshold)
        runs.append(tmp[0])
        rdys = tmp[1]
        progs = tmp[2]
        plott = tmp[3]
    return [runs,rdys,progs,plott]

def game_loops(x, noise, shuffle, info_vis, info_clust, rdys, progs, plott, noisyness, threshold):
    graf = Group(players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold)
    graf.update_cs()
    plott.append([graf.progress()])
    time = rounds
    for i in range(1,rounds+1):
        if not graf.check_end():
            #graf.threshold = i/rounds
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

        else:
            if time == rounds:
                time = i
                rdys.append(i)
        plott[x].append(graf.progress())
    progs.append(graf.progress())
    return [time, rdys, progs, plott]

def noise_onoff(n_games, t1, namount):
    games = n_games
    shuffle = False
    info_vis = True
    info_clust = False
    noise = False
    if namount ==0:
        noise=False
    game1 = game(noise, games,shuffle,info_vis, info_clust, namount, t1)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    if len(rdys)==0:
        rdys.append(0)
    print(" RESULTS (BOT AI NOISE off/on, Random:", shuffle,", Info (Vision/CS): ", info_vis, "/", info_clust,")")
    print("AVG ROUNDS: ", np.average(runs))
    print("COMPLETIONS: ", progs.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs))
    print("VAR PROGRESS: ", np.var(progs))
    print("MAX PROGRESS: ", max(progs))
    print("MIN PROGRESS: ", min(progs))
    print("GOALTIME ", np.average(rdys))
    paramlog.append([namount, np.average(progs)])
    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    #plt.figure("Noise on/off")
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "'AI NOISE '"+str(namount))
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
    shuffle = False
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

def clust_param(n_games, t1):
    games = n_games
    shuffle = True
    info_vis = True
    noise = False
    info_clust = True
    game1 = game(noise, games,shuffle,False, True, 0, t1)
    runs = game1[0]
    rdys = game1[1]
    progs = game1[2]
    plott = game1[3]
    if len(rdys)==0:
        rdys.append(0)
    print(" RESULTS (THRESHOLD", t1, "Random:", shuffle,", Noise: ", noise, ")")
    print("AVG ROUNDS: ", np.average(runs))
    print("COMPLETIONS: ", progs.count(1.0)/games)
    print("AVG PROGRESS: ", np.average(progs))
    print("VAR PROGRESS: ", np.var(progs))
    print("MAX PROGRESS: ", max(progs))
    print("MIN PROGRESS: ", min(progs))
    print("GOALTIME ", np.average(rdys))

    paramlog.append([t1, np.average(progs)])

    #PLOTTING
    lst = np.asarray(plott).T.tolist()
    #plt.figure("Cluster information on/off")
    avgs = []
    for j in range(0, len(lst)):
        avgs.append(np.average(lst[j]))
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = "AI threshold "+str(t1))
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')

plt.figure("Progress")


#clust_onoff(100)
#noise_amount(100)
#noise_onoff(100)
paramlog = []
noise_onoff(100, 1, 0)

def plotparam():
    params = np.arange(0,60,10)
    for t in params:
        noise_onoff(100, 0.4, t)
        #clust_param(100, t)
    plt.figure("Progress with noises")
    lst = np.asarray(paramlog).T.tolist()
    plt.plot(lst[0], lst[1], label = "Threshold progress "+str(rounds))
    plt.xlabel('Threshold')
    plt.ylabel('Average progress')
    plt.axis((0,50,0,1))
    #for xy in paramlog:
    #    plt.annotate('(%s, %s)' % (round(xy[0],2), round(xy[1],2)), xy=xy, textcoords='data')
    plt.legend(loc='best')

#plotparam()
'''ns = np.arange(20,80,10)
for s in ns:
    paramlog = []
    players = s
    plotparam()'''

plt.ion()
plt.show()
input()
#CSV writing/parsing etc
