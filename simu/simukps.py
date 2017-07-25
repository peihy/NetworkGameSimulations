import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import networkx as nx
from operator import itemgetter
import time

'''
Initial setup
'''
rounds = 50
players = 36
ncolors = 3
shrt = 3
color_names = [1, 2, 3]
drawcolors = ['r', 'g', 'b']
gametype = "Consensus"

colors=[]
for i in range(players):
    colors.append(color_names[i%ncolors])
#random.shuffle(colors)
def initialforms():
    B = nx.cycle_graph(players)
    B = nx.convert_node_labels_to_integers(B, first_label = 1, ordering="sorted")
    b_pos = nx.circular_layout(B)
    for i in range(1,players+1):
        if i < players-1:
            B.add_edge(i,i+2)
        elif i == players:
            B.add_edge(i, 2)
        else:
            B.add_edge(i, players-i)
    plt.figure("Initial regular")
    z=1
    for color in colors:
        nx.draw_networkx_nodes(B, b_pos, nodelist=[z], node_color=drawcolors[color-1])
        z+=1
    nx.draw_networkx_edges(B,b_pos,width=1.0,alpha=0.5)

    #A = nx.watts_strogatz_graph(players, 4, 0.3, seed=None)
    A = nx.cycle_graph(players)
    A = nx.convert_node_labels_to_integers(A, first_label = 1, ordering="sorted")
    a_pos = nx.circular_layout(A)
    shorts1, shorts2, shorts3, x = [], [],[], 1
    for i in range(1,players+1):
        if i == x*(players/shrt)-2:
            shorts1.append(i)
            shorts2.append(i-2)
            shorts3.append(i+2)
            x+=1
    x=0
    for i in range(1,players+1):
        if i not in shorts2:
            if i < players-1:
                A.add_edge(i,i+2)
            elif i == players:
                A.add_edge(i, 2)
            else:
                A.add_edge(i, players-i)
    '''for i in shorts1:
        for j in shorts1:
            if i != j:
                A.add_edge(i, j)'''
    A.add_edge(shorts1[0], shorts2[len(shorts2)-1])
    for i in range(len(shorts2)-1):
        A.add_edge(shorts2[i], int(shorts1[i+1]))

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
    middle = float(len(tmp))/2
    if middle % 2 != 0:
        d1=tmp[int(middle - .5)]
        d2=int(players/d1)
    else:
        d1=tmp[int(middle-1)]
        d2=int(players/d1)

    F = nx.grid_graph(dim=[int(d1), int(d2)])
    F = nx.convert_node_labels_to_integers(F, first_label = 1, ordering="sorted")

    square = d1
    pos, i, row = {}, 1, 0
    for y in range(d1):
        for x in range(d2):
            pos[i] = [x*50, y*50]
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
    if True:
        random.shuffle(right)
        random.shuffle(left)
        random.shuffle(bot)
        random.shuffle(top)
    for r in range(len(right)):
        F.add_edge(left[r], right[r])
    for r in range(len(top)):
        F.add_edge(top[r], bot[r])

    plt.figure("Initial mesh")
    z=1
    for color in colors:
        nx.draw_networkx_nodes(F, pos, nodelist=[z], node_color=drawcolors[color-1])
        z+=1
    nx.draw_networkx_edges(F,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_labels(F,pos,width=1.0,alpha=0.5)
    print(nx.average_shortest_path_length(B))
    print(nx.average_shortest_path_length(A))
    print(nx.average_shortest_path_length(F))


initialforms()
#print(nx.average_degree_connectivity(A))
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
        self.points = 0

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
                cs = 100
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
            cs = 100
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
        for i in range(players):
            self.colors.append(color_names[i%ncolors])
        if shuffle:
            random.shuffle(self.colors)

        #self.network = nx.convert_node_labels_to_integers(nx.grid_graph(dim=[4, 4]), first_label = 1, ordering="sorted")
        if net == "B":
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

        elif net == "A":
            R = nx.cycle_graph(players)
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")
            pos = nx.circular_layout(R)
            shorts1, shorts2, shorts3, x = [], [],[], 1
            for i in range(1,players+1):
                if i == x*(players/shrt)-2:
                    shorts1.append(i)
                    shorts2.append(i-2)
                    shorts3.append(i+2)
                    x+=1
            x=0
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
            self.pos = nx.circular_layout(R)
        elif net == "C":
            d1=players-1
            tmp=[]
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
            R = nx.convert_node_labels_to_integers(R, first_label = 1, ordering="sorted")

            square = d1
            pos, i, row = {}, 1, 0
            for y in range(d1):
                for x in range(d2):
                    pos[i] = [x*50, y*50]
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
            if edgeshuffle:
                random.shuffle(right)
                random.shuffle(left)
                random.shuffle(bot)
                random.shuffle(top)
            for r in range(len(right)):
                R.add_edge(left[r], right[r])
            for r in range(len(top)):
                R.add_edge(top[r], bot[r])
            self.pos = pos
        else:
            R = nx.random_regular_graph(3, players)
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
                p.points = 0
                if clr==p.color:
                    ii = p.id
                    G.add_node(ii)
                    for n in nx.all_neighbors(self.network, ii):
                        neib = self.get_player_by_id(n)
                        if neib.color == clr:
                            p.points +=1
                            jj = neib.id
                            G.add_edge(ii,jj)
                        else: p.points -=1
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

        for color in colors:
            tmp = []
            for p in self.players:
                clrs_lst.append(p.cluster_size)
                if p.color == color: tmp.append(p.cluster_size)
            color_lst.append(max(tmp))
            max_clrs.append(len(tmp))

        if gametype == "Consensus":
            avgsum = 0
            psum = 0
            for i in range(ncolors):
                avgsum += (color_lst[i]/max_clrs[i])
            for p in self.players:
                psum += p.points
            return [avgsum/ncolors, psum/players]
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
        pos = {}
        plt.figure("End of game: "+str(name)+ " "+str(prog))
        z=1
        for color in self.colors:
            nx.draw_networkx_nodes(self.network, self.pos, nodelist=[z], node_color=drawcolors[color-1])
            z+=1
        nx.draw_networkx_edges(self.network,self.pos,width=1.0,alpha=0.5)
        nx.draw_networkx_labels(self.network,self.pos,width=1.0,alpha=0.5)

        #plt.show()
        #time.sleep(2)


#Gameloop

def game(ai, games, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle):
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
    graf = Group(players, colors, noise, shuffle, info_vis, info_clust, noisyness, threshold, net, edgeshuffle)
    graf.update_cs()
    plott.append([graf.progress()[0]])
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
        plott[x].append(graf.progress()[0])
    #graf.print_graph(x, graf.progress())
    progs.append(graf.progress()[0])
    print(graf.progress()[1])
    if graf.progress()[0] > 0.6:
        rdys.append(1)
    return [time, rdys, progs, plott, nx.average_shortest_path_length(graf.network)]

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
    plt.plot(np.arange(1,len(avgs)+1), avgs, label = str(noisyness) + "% noise, AI threshold "+str(t1)+name)
    plt.xlabel('Round n')
    plt.ylabel('Average progress')
    plt.legend(loc='best')

#plt.figure("Progress")

#clust_onoff(100)
#noise_amount(100)
#noise_onoff(100)
paramlog = []
#clust_param(100, 1, "A", " smallworld, noise", True, 100)
#clust_param(100, 1, "A", " smallworld", False, 100)
#clust_param(100, 1, "A", " smallworld", 50)
#clust_param(100, 1, "A", " smallworld", 40)
#clust_param(100, 1, "A", " smallworld", 30)
#clust_param(100, 1, "A", " smallworld", 0)

#clust_param(50, 1, "C", " mesh noise: ", False, True, 100)
clust_param(20, 0.55, "D", False, " random ", True, False, 10)
clust_param(20, 0.55, "C", True, " mesh shuf ", True, False, 10)
clust_param(20, 0.55, "C", False, " mesh ", True, False, 10)
clust_param(20, 0.55, "B", False, " ring ", True, False, 10)
clust_param(20, 0.55, "A", False, " sw ", True, False, 10)

#clust_param(20, 0.4, "B", " reg  ", True, False, 20)
#clust_param(20, 0.4, "A", " sw  ", True, False, 20)

#clust_param(50, 1, "B", " reg noise: ", False, True, 100)
#clust_param(1, 0.3, "B", " reg clust info: ", True, False, 10)

#clust_param(100, 1, "B", " regular")

def plotparam():
    params = np.arange(0,110,10)
    #params = np.arange(0.2,1.1,0.1)

    for t in params:
        #clust_param(50, 1, "A", " noise: "+str(t), False, True, t)
        clust_param(50, 0.6, "A", " with "+str(t), True, False, t)

    plt.figure("Progress with noises")
    lst = np.asarray(paramlog).T.tolist()
    plt.plot(lst[0], lst[1], label = "Threshold progress "+str(rounds))
    plt.xlabel('Threshold')
    plt.ylabel('Average progress')
    plt.axis((0,100,0,1))
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
