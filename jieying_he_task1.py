from pyspark import SparkConf, SparkContext
import time
import random
import sys
start = time.time()

def generateRandom(num):
    if num == 1:
        return -1
    else:
        return 1

if __name__ == "__main__":
    conf = SparkConf().setAppName("cvsApp").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    #csv_path = '../hw3dataset/yelp_train.csv'
    #output_path = "task1_2.txt"
    #output_path = '1.txt'
    #output_path1 = '2.txt'
    #output_path2 = '3.txt'
    #output_path3 = '4.txt'

    csv_path = sys.argv[1]
    similarity_method = sys.argv[2]
    output_path = sys.argv[3]

    if similarity_method == "jaccard":
        csvRDD = sc.textFile(csv_path)
        head = csvRDD.first()
        users = csvRDD.filter(lambda row: row != head).map(lambda row: row.split(",")[0]).distinct().persist()
        user_count = users.count()
        #print(user_count)
        user_map = {}
        for index, user in enumerate(users.collect()):
            user_map[user] = index
        businesses = csvRDD.filter(lambda row: row != head).map(lambda row: row.split(",")[1]).distinct().persist()
        businesses_num = businesses.count()
        #print(businesses_num)
        bus_id_map = {}
        id_bus_map = {}
        for index, business in enumerate(businesses.collect()):
            bus_id_map[business] = index
            id_bus_map[index] = business
        #print(bus_id_map)
        #print(id_bus_map)
        data_set = csvRDD.filter(lambda row: row != head).map(lambda row: (row.split(",")[1], row.split(",")[0])).groupByKey().persist()
        #print(user_map)
        business_num = data_set.count()
        #print(business_num)

        all_data = data_set.collect()
        sig_matrix = [[0 for j in range(business_num)]for i in range(100)]
        for i in range(100):
            a, b = random.randint(1, 100000), random.randint(1, 100000)
            for data in all_data:
                each_business = []
                for user in data[1]:
                    x = user_map[user]
                    hash_num = (a * x + b) % business_num
                    each_business.append(hash_num)
                min_num = min(each_business)
                sig_matrix[i][bus_id_map[data[0]]] = min_num

        band = 50
        row = 2
        k = business_num * 100
        buckets = {}
        r = 0
        for i in range(band):
            b_k = {}
            for b in range(business_num):
                col = (sig_matrix[r][b] * 1 + sig_matrix[r + 1][b] * 2) % k
                if col not in b_k:
                    b_k[col] = []
                b_k[col].append(b)
            buckets[i] = b_k
            r = r + row

        data_dict = {}
        for data in all_data:
            user_set = set(data[1])
            data_dict[data[0]] = user_set
        J_Sim = {}
        for i in range(band):
            bucket = buckets[i]
            for key, value in bucket.items():
                if len(bucket[key]) >= 2:
                    bin = bucket[key]
                    for a in range(len(bin) - 1):
                        for b in range(a + 1, len(bin)):
                            s = []
                            s.append(id_bus_map[bin[a]])
                            s.append(id_bus_map[bin[b]])
                            s.sort()
                            s = frozenset(s)
                            sim = len(data_dict[id_bus_map[bin[a]]] & data_dict[id_bus_map[bin[b]]]) / len(data_dict[id_bus_map[bin[a]]] | data_dict[id_bus_map[bin[b]]])
                            if s not in J_Sim and sim >= 0.5:
                                J_Sim[s] = sim
        with open(output_path, 'w') as file:
            #print(len(J_Sim))
            file.write("business_id_1, business_id_2, similarity\n")
            sim_can = [[]for i in range(len(J_Sim))]
            i = 0
            for key, value in J_Sim.items():
                key = list(key)
                key.sort()
                sim_can[i].append(key)
                sim_can[i].append(value)
                i += 1
            sim_can.sort(key=lambda row: (row[0][0], row[0][1]))
            for sim in sim_can:
                file.write(str(sim[0][0]))
                file.write(', ')
                file.write(str(sim[0][1]))
                file.write(', ')
                file.write(str(sim[1]))
                file.write('\n')

    '''with open(output_path, 'w') as file:
        for data in all_data:
            file.write('(')
            file.write(data[0])
            file.write(',(')
            for d in data[1]:
                file.write(d)
                file.write(',')
            file.write(')')
            file.write(')\n')
    #print(data_set)
    with open(output_path1, 'w') as file:
        for i in range(100):
            for j in range(business_num):
                file.write(str(sig_matrix[i][j]))
                file.write(',')
            file.write('\n')

    with open(output_path2, 'w') as file:
        for i in range(band):
            bucket = buckets[i]
            for key, value in bucket.items():
                if len(bucket[key]) >= 2:
                    file.write('(')
                    for b in bucket[key]:
                        file.write(str(b))
                        file.write(',')
                    file.write(')')
            file.write('\n')'''
    if similarity_method == "cosine":
        csvRDD = sc.textFile(csv_path)
        head = csvRDD.first()
        users = csvRDD.filter(lambda row: row != head).map(lambda row: row.split(",")[0]).distinct().persist()
        user_count = users.count()
        # print(user_count)
        user_map = {}
        for index, user in enumerate(users.collect()):
            user_map[user] = index
        businesses = csvRDD.filter(lambda row: row != head).map(lambda row: row.split(",")[1]).distinct().persist()
        businesses_num = businesses.count()
        # print(businesses_num)
        bus_id_map = {}
        id_bus_map = {}
        for index, business in enumerate(businesses.collect()):
            bus_id_map[business] = index
            id_bus_map[index] = business
        #data_set = csvRDD.filter(lambda row: row != head).map(lambda row: (row.split(",")[1], row.split(",")[0])).groupByKey().persist()
        # print(bus_id_map)
        # print(id_bus_map)
        # data_set = data_set.collect()
        # print(user_map)
        #business_num = data_set.count()
        #print(business_num)
        item_user_data = csvRDD.filter(lambda row: row != head).map(
            lambda row: (row.split(",")[1], (row.split(",")[0], row.split(",")[2]))).groupByKey().persist()
        business_num = item_user_data.count()
        print(business_num)
        item_user_dict = {}
        for item in item_user_data.collect():
            rating = {}
            for user in item[1]:
                rating[user[0]] = float(user[1])
            item_user_dict[item[0]] = rating

        all_data = item_user_data.collect()
        sig_matrix = [[0 for j in range(business_num)] for i in range(400)]
        v = {}
        for i in range(400):
            for user in users.collect():
                v[user] = generateRandom(random.randint(1, 2))
            for data in all_data:
                sum = 0
                for user in data[1]:
                   sum += item_user_dict[data[0]][user[0]] * v[user[0]]
                if sum >= 0:
                    sig_matrix[i][bus_id_map[data[0]]] = 1
                else:
                    sig_matrix[i][bus_id_map[data[0]]] = -1
        '''output_path1 = "test100.txt"
        with open(output_path1, 'w') as file:
            for i in range(400):
                for j in range(business_num):
                    file.write(str(sig_matrix[i][j]))
                    file.write(',')
                file.write('\n')'''
        band = 20
        row = 20
        buckets = {}
        r = 0
        for i in range(band):
            b_k = {}
            for b in range(business_num):
                col = str(i) + str("#") + str(sig_matrix[r][b]) + str(sig_matrix[r + 1][b]) + str(sig_matrix[r + 2][b]) + str(sig_matrix[r + 3][b]) + str(sig_matrix[r + 4][b]) + str(sig_matrix[r + 5][b]) + str(sig_matrix[r + 6][b]) + str(sig_matrix[r + 7][b]) + str(sig_matrix[r + 8][b]) + str(sig_matrix[r + 9][b]) + str(sig_matrix[r + 10][b]) + str(sig_matrix[r + 11][b]) + str(sig_matrix[r + 12][b]) + str(sig_matrix[r + 13][b]) + str(sig_matrix[r + 14][b]) + str(sig_matrix[r + 15][b]) + str(sig_matrix[r + 16][b]) + str(sig_matrix[r + 17][b]) + str(sig_matrix[r + 18][b]) + str(sig_matrix[r + 19][b])
                if col not in b_k:
                    b_k[col] = []
                b_k[col].append(b)
            buckets[i] = b_k
            r = r + row
        C_Sim = {}
        for i in range(band):
            bucket = buckets[i]
            for key, value in bucket.items():
                if len(bucket[key]) >= 2:
                    bin = bucket[key]
                    for a in range(len(bin) - 1):
                        u_set = set(item_user_dict[id_bus_map[bin[a]]].keys())
                        for b in range(a + 1, len(bin)):
                            s = []
                            s.append(id_bus_map[bin[a]])
                            s.append(id_bus_map[bin[b]])
                            s.sort()
                            s = frozenset(s)
                            v_set = set(item_user_dict[id_bus_map[bin[b]]].keys())
                            c_set = u_set & v_set
                            m = 0
                            print(1)
                            for c in c_set:
                                m += item_user_dict[id_bus_map[bin[a]]][c] * item_user_dict[id_bus_map[bin[b]]][c]
                            n1 = 0
                            print(2)
                            for u in u_set:
                                n1 += item_user_dict[id_bus_map[bin[a]]][u] ** 2
                            n1 = n1 ** 0.5
                            n2 = 0
                            print(3)
                            for v in v_set:
                                n2 += item_user_dict[id_bus_map[bin[b]]][v] ** 2
                            n2 = n2 ** 0.5
                            n = n1 * n2
                            sim = m / n
                            if s not in C_Sim and sim >= 0.5:
                                C_Sim[s] = sim
        with open(output_path, 'w') as file:
            # print(len(J_Sim))
            file.write("business_id_1, business_id_2, similarity\n")
            sim_can = [[] for i in range(len(C_Sim))]
            i = 0
            for key, value in C_Sim.items():
                key = list(key)
                key.sort()
                sim_can[i].append(key)
                sim_can[i].append(value)
                i += 1
            sim_can.sort(key=lambda row: (row[0][0], row[0][1]))
            for sim in sim_can:
                file.write(str(sim[0][0]))
                file.write(', ')
                file.write(str(sim[0][1]))
                file.write(', ')
                file.write(str(sim[1]))
                file.write('\n')

    end = time.time()
    time = end - start
    print("Duration:" + str(time))
