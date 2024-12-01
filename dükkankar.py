import numpy as np
import matplotlib.pyplot as plt


def load_data():
    x = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598])
    y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])
    
    return x,y

x_train,y_train=load_data()


def cost_function(x, y, w, b):
    
    m=x.shape[0]
    total_cost=0
    for i in range(m):
        f_wb=((w*x[i])+b)
        total_cost+=(f_wb-y[i])**2
    total_cost=total_cost/(2*m)
    return total_cost
    

def cost_türev(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb = (w * x[i]) + b
        dj_dw += (f_wb - y[i])*x[i]
        dj_db += (f_wb - y[i])
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db


def alphahesap(x,y,w_in, b_in, num_iters,cost_türevi,totalhesap,alpha):
    m=x.shape[0]
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw,dj_db=cost_türevi(x,y,w,b)
        w-=(alpha*dj_dw)
        b-=(alpha*dj_db)
    return w,b

iteration=15000
w_initial=0
b_initial=0
alpha=0.01

w, b = alphahesap(x_train, y_train, w_initial, b_initial, iteration, cost_türev, cost_function, alpha)

m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = (w * x_train[i]) + b

plt.scatter(x_train, y_train, marker="x", c="r", label="Training Data")
plt.plot(x_train, predicted, c="b", label="Prediction")
plt.title("Konuma göre Kâr")
plt.xlabel("Kira")
plt.ylabel("Kâr")
plt.legend()
plt.show()


nüfus=int(input("Küçük Şehrin nüfusunu milyon cinsinden giriniz: "))
nüfus2=int(input("Büyük Şehrin nüfusunu milyon cinsinden giriniz: "))

predict1 = nüfus * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = nüfus2 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))
        
        



        