from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkConf, SparkContext
import time
import csv
import collections
import random
import sys
start = time.time()

if __name__ == "__main__":
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    case_id = int(sys.argv[3])
    output_file_path = sys.argv[4]
    conf = SparkConf().setAppName("cvsApp").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    #train_file_path = '../hw3dataset/yelp_train.csv'
    #test_file_path = '../hw3dataset/yelp_val.csv'
    #output_file_path = 'task2_1.csv'
    #output_file_path1 = 'task2_2.csv'
    #output_file_path2 = 'task2_3.csv'
    #output_file_path3 = 'task2_4.csv'
    #output_path1 = 'test.txt'
    #output_path2 = 'test1.txt'
    #output_path3 = 'test2.txt'
    #output_path4 = 'test3.txt'

    trainRDD = sc.textFile(train_file_path)
    train_head = trainRDD.first()
    testRDD = sc.textFile(test_file_path)
    test_head = testRDD.first()

    if case_id == 1:
        users = trainRDD.filter(lambda row: row != train_head).map(lambda row: row.split(",")[0]).distinct().persist()
        user_map = collections.defaultdict(int)
        id_user_map = {}
        for index, user in enumerate(users.collect()):
            user_map[user] = index
            id_user_map[index] = user
        businesses = trainRDD.filter(lambda row: row != train_head).map(lambda row: row.split(",")[1]).distinct().persist()
        bus_id_map = collections.defaultdict(int)
        id_bus_map = {}
        for index, business in enumerate(businesses.collect()):
            bus_id_map[business] = index
            id_bus_map[index] = business
        #print(bus_id_map)
        #print(id_bus_map)

        #case1:
        # Load and parse the data
        train_ratings = trainRDD.filter(lambda row: row != train_head).map(lambda row: row.split(',')).map(lambda row: Rating(user_map[row[0]], bus_id_map[row[1]], float(row[2])))

        # Build the recommendation model using Alternating Least Squares
        rank = 10
        numIterations = 10
        model = ALS.train(train_ratings, rank, numIterations, lambda_=0.25)

        # Evaluate the model on training data
        #test_file_path = '../hw3dataset/yelp_val.csv'
        test_ratings = testRDD.filter(lambda row: row != test_head).map(lambda row: row.split(',')).map(lambda row: Rating(user_map[row[0]], bus_id_map[row[1]], float(row[2])))
        testdata = test_ratings.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        #ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        #MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        #print("Mean Squared Error = " + str(MSE))


        with open(output_file_path, "w") as csvFile:
            out = csv.writer(csvFile)
            out.writerow(['user_id', 'business_id', 'prediction'])
            for pred in predictions.collect():
                out.writerow([id_user_map[pred[0][0]], id_bus_map[pred[0][1]], pred[1]])

        # Save and load model
        #model.save(sc, "target/tmp/myCollaborativeFilter")
        #sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

    if case_id == 2:
        #case2:
        userCF_data = trainRDD.filter(lambda row: row != train_head).map(lambda row: (row.split(",")[0], (row.split(",")[1], row.split(",")[2]))).groupByKey().persist()
        bus_dataset = trainRDD.filter(lambda row: row != train_head).map(lambda row: (row.split(",")[1], row.split(",")[0])).groupByKey().persist()
        test_data = testRDD.filter(lambda row: row != test_head).map(lambda row: (row.split(",")[0], row.split(",")[1])).persist()
        '''with open(output_path1, 'w') as file:
            for data in userCF_data.collect():
                file.write('(')
                file.write(data[0])
                file.write(',(')
                for d in data[1]:
                    file.write(d[0])
                    file.write(':')
                    file.write(str(d[1]))
                    file.write(',')
                file.write('),')
                file.write(')\n')'''
        avg_user = collections.defaultdict(float)
        for user in userCF_data.collect():
            item_count = len(user[1])
            item = {}
            for i in user[1]:
                item[i[0]] = float(i[1])
            rating_sum = sum(item.values())
            avg_user[user[0]] = rating_sum / item_count
        #print(avg_user)
        bus_dict = {}
        for bus in bus_dataset.collect():
            bus_dict[bus[0]] = []
            for user in bus[1]:
                bus_dict[bus[0]].append(user)
        user_item_dict = {}
        for user in userCF_data.collect():
            rating = {}
            for item in user[1]:
                rating[item[0]] = float(item[1])
            user_item_dict[user[0]] = rating
        '''with open(output_path2, 'w') as file:
            for key, value in bus_dict.items():
                file.write('(')
                file.write(key)
                file.write(',(')
                for v in value:
                    file.write(v)
                    file.write(',')
                file.write('),')
                file.write(')\n')
        with open(output_path3, 'w') as file:
            for key, value in user_item_dict.items():
                file.write('(')
                file.write(key)
                file.write(',(')
                for k, v in user_item_dict[key].items():
                    file.write(k)
                    file.write(':')
                    file.write(str(v))
                    file.write(',')
                file.write(')')
                file.write(')\n')'''
        P = {}
        W = {}
        for row in test_data.collect():
            user = row[0]
            bus = row[1]
            avg = avg_user[user]
            w = {}
            if user in user_item_dict:
                if bus in bus_dict:
                    u_set = set(user_item_dict[user].keys())
                    for u in bus_dict[bus]:
                        u_pair = []
                        u_pair.append(user)
                        u_pair.append(u)
                        k = frozenset(u_pair)
                        if k in W:
                            w[k] = W[k]
                        if k not in W:
                            v_set = set(user_item_dict[u].keys())
                            r = u_set & v_set
                            a = 0
                            for i in r:
                                a += (user_item_dict[user][i] - avg) * (user_item_dict[u][i] - avg_user[u])
                            b1 = 0
                            for i in r:
                                b1 += (user_item_dict[user][i] - avg) ** 2
                            b1 = b1 ** 0.5
                            b2 = 0
                            for i in r:
                                b2 += (user_item_dict[u][i] - avg_user[u]) ** 2
                            b2 = b2 ** 0.5
                            b = b1 * b2
                            if b == 0:
                                W[k] = 0
                                w[k] = 0
                            else:
                                W[k] = a / b
                                w[k] = a / b
                    m = 0
                    for u in bus_dict[bus]:
                        u_pair = []
                        u_pair.append(user)
                        u_pair.append(u)
                        k = frozenset(u_pair)
                        m += (user_item_dict[u][bus] - avg_user[u]) * w[k]
                    n = 0
                    for k, v in w.items():
                        n += abs(w[k])
                    if n == 0:
                        P[(user, bus)] = avg
                    else:
                        P[(user, bus)] = avg + (m / n)
                elif bus not in bus_dict:
                    P[(user, bus)] = avg
            elif user not in user_item_dict:
                if bus in bus_dict:
                    res = 0
                    count = len(bus_dict[bus])
                    for u in bus_dict[bus]:
                        res += user_item_dict[u][bus]
                    res = res / count
                    P[(user, bus)] = res
                elif bus not in bus_dict:
                    P[(user, item)] = 0
        with open(output_file_path, "w") as csvFile:
            out = csv.writer(csvFile)
            out.writerow(['user_id', 'business_id', 'prediction'])
            for pred in P:
                out.writerow([pred[0], pred[1], P[pred]])

    if case_id == 3:
        #case3.1:
        users = trainRDD.filter(lambda row: row != train_head).map(lambda row: row.split(",")[0]).distinct().persist()
        user_map = collections.defaultdict(int)
        for index, user in enumerate(users.collect()):
            user_map[user] = index
        businesses = trainRDD.filter(lambda row: row != train_head).map(
            lambda row: row.split(",")[1]).distinct().persist()
        bus_id_map = collections.defaultdict(int)
        id_bus_map = {}
        for index, business in enumerate(businesses.collect()):
            bus_id_map[business] = index
            id_bus_map[index] = business
        data_set = trainRDD.filter(lambda row: row != train_head).map(
            lambda row: (row.split(",")[1], row.split(",")[0])).groupByKey().persist()
        # print(user_map)
        business_num = data_set.count()
        #print(business_num)
        all_data = data_set.collect()
        sig_matrix = [[0 for j in range(business_num)] for i in range(100)]
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
                            p = []
                            p.append(a)
                            p.append(b)
                            p = frozenset(p)
                            J_Sim[p] = 1
        itemCF_data = trainRDD.filter(lambda row: row != train_head).map(
            lambda row: (row.split(",")[1], (row.split(",")[0], row.split(",")[2]))).groupByKey().persist()
        user_dataset = trainRDD.filter(lambda row: row != train_head).map(
            lambda row: (row.split(",")[0], row.split(",")[1])).groupByKey().persist()
        test_data = testRDD.filter(lambda row: row != test_head).map(
            lambda row: (row.split(",")[0], row.split(",")[1])).persist()
        avg_item = collections.defaultdict(float)
        for item in itemCF_data.collect():
            user_count = len(item[1])
            user = {}
            for u in item[1]:
                user[u[0]] = float(u[1])
            rating_sum = sum(user.values())
            avg_item[item[0]] = rating_sum / user_count
        #print(avg_item)
        user_dict = {}
        for user in user_dataset.collect():
            user_dict[user[0]] = []
            for item in user[1]:
                user_dict[user[0]].append(item)
        item_user_dict = {}
        for item in itemCF_data.collect():
            rating = collections.defaultdict(float)
            for user in item[1]:
                rating[user[0]] = float(user[1])
            item_user_dict[item[0]] = rating
        P = {}
        W = {}
        for row in test_data.collect():
            user = row[0]
            item = row[1]
            avg = avg_item[item]
            w = {}
            if item in item_user_dict:
                if user in user_dict:
                    u_set = set(item_user_dict[item].keys())
                    for i in user_dict[user]:
                        i_pair = []
                        i_pair.append(item)
                        i_pair.append(i)
                        k = frozenset(i_pair)
                        if k in J_Sim:
                            if k in W:
                                w[k] = W[k]
                            if k not in W:
                                v_set = set(item_user_dict[i].keys())
                                r = u_set & v_set
                                a = 0
                                for u in r:
                                    a += (item_user_dict[item][u] - avg) * (item_user_dict[i][u] - avg_item[i])
                                b1 = 0
                                for u in r:
                                    b1 += (item_user_dict[item][u] - avg) ** 2
                                b1 = b1 ** 0.5
                                b2 = 0
                                for u in r:
                                    b2 += (item_user_dict[i][u] - avg_item[i]) ** 2
                                b2 = b2 ** 0.5
                                b = b1 * b2
                                if b == 0:
                                    W[k] = 0
                                    w[k] = 0
                                else:
                                    W[k] = a / b
                                    w[k] = a / b
                    m = 0
                    for i in user_dict[user]:
                        i_pair = []
                        i_pair.append(item)
                        i_pair.append(i)
                        k = frozenset(i_pair)
                        if k in J_Sim:
                            m += (item_user_dict[i][user] - avg_item[i]) * w[k]
                    n = 0
                    for k, v in w.items():
                        n += abs(w[k])
                    if n == 0:
                        P[(user, item)] = avg
                    else:
                        P[(user, item)] = avg + (m / n)
                elif user not in user_dict:
                    P[(user, item)] = avg
            elif item not in item_user_dict:
                if user in user_dict:
                    res = 0
                    count = len(user_dict[user])
                    for i in user_dict[user]:
                        res += item_user_dict[i][user]
                    res = res / count
                    P[(user, item)] = res
                elif user not in user_dict:
                    P[(user, item)] = 0
        with open(output_file_path, "w") as csvFile:
            out = csv.writer(csvFile)
            out.writerow(['user_id', 'business_id', 'prediction'])
            for pred in P:
                out.writerow([pred[0], pred[1], P[pred]])

    '''if case_id == 4:
        itemCF_data = trainRDD.filter(lambda row: row != train_head).map(
            lambda row: (row.split(",")[1], (row.split(",")[0], row.split(",")[2]))).groupByKey().persist()
        user_dataset = trainRDD.filter(lambda row: row != train_head).map(
            lambda row: (row.split(",")[0], row.split(",")[1])).groupByKey().persist()
        test_data = testRDD.filter(lambda row: row != test_head).map(
            lambda row: (row.split(",")[0], row.split(",")[1])).persist()
        avg_item = collections.defaultdict(float)
        for item in itemCF_data.collect():
            user_count = len(item[1])
            user = {}
            for u in item[1]:
                user[u[0]] = float(u[1])
            rating_sum = sum(user.values())
            avg_item[item[0]] = rating_sum / user_count
        #print(avg_item)
        user_dict = {}
        for user in user_dataset.collect():
            user_dict[user[0]] = []
            for item in user[1]:
                user_dict[user[0]].append(item)
        item_user_dict = {}
        for item in itemCF_data.collect():
            rating = {}
            for user in item[1]:
                rating[user[0]] = float(user[1])
            item_user_dict[item[0]] = rating
        P = {}
        W = {}
        for row in test_data.collect():
            user = row[0]
            item = row[1]
            avg = avg_item[item]
            w = {}
            if item in item_user_dict:
                if user in user_dict:
                    u_set = set(item_user_dict[item].keys())
                    for i in user_dict[user]:
                        i_pair = []
                        i_pair.append(item)
                        i_pair.append(i)
                        k = frozenset(i_pair)
                        if k in W:
                            w[k] = W[k]
                        if k not in W:
                            v_set = set(item_user_dict[i].keys())
                            r = u_set & v_set
                            a = 0
                            for u in r:
                                a += (item_user_dict[item][u] - avg) * (item_user_dict[i][u] - avg_item[i])
                            b1 = 0
                            for u in r:
                                b1 += (item_user_dict[item][u] - avg) ** 2
                            b1 = b1 ** 0.5
                            b2 = 0
                            for u in r:
                                b2 += (item_user_dict[i][u] - avg_item[i]) ** 2
                            b2 = b2 ** 0.5
                            b = b1 * b2
                            if b == 0:
                                W[k] = 0
                                w[k] = 0
                            else:
                                W[k] = a / b
                                w[k] = a / b
                    m = 0
                    for i in user_dict[user]:
                        i_pair = []
                        i_pair.append(item)
                        i_pair.append(i)
                        k = frozenset(i_pair)
                        m += (item_user_dict[i][user] - avg_item[i]) * w[k]
                    n = 0
                    for k, v in w.items():
                        n += abs(w[k])
                    if n == 0:
                        P[(user, item)] = avg
                    else:
                        P[(user, item)] = avg + (m / n)
                elif user not in user_dict:
                    P[(user, item)] = avg
            elif item not in item_user_dict:
                if user in user_dict:
                    res = 0
                    count = len(user_dict[user])
                    for i in user_dict[user]:
                        res += item_user_dict[i][user]
                    res = res / count
                    P[(user, item)] = res
                elif user not in user_dict:
                    P[(user, item)] = 0
        with open(output_file_path, "w") as csvFile:
            out = csv.writer(csvFile)
            out.writerow(['user_id', 'business_id', 'prediction'])
            for pred in P:
                out.writerow([pred[0], pred[1], P[pred]])'''

    end = time.time()
    time = end - start
    print("Duration:" + str(time))