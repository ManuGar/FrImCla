from multiprocessing import Pool

def f(pair):
    (x,y)=pair
    return x*y

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [(1,4),(2,5),(3,6)]))
