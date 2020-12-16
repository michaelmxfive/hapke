import numpy as np
import sympy as sp
import copy
import math
import random
import csv
import scipy.optimize as optimize
from nelder_mead import NelderMead
from scipy.optimize import minimize

#	--- input ---
	#	i: incidence angle in radian
	#	e: emission angle in radian
	#	a: phase angle in radian
	#	w: single-scattering albedo
	#	g: asymmetry parameter
	#	B0: amplitude of opposition effect
	#	h: opposition surge angular width
	#	t: roughness parameter in radian
	#	--- output ---
	#	r: radiance coefficient, as the brightness of a surface relative to
	#   the brightness of a Lambert surface identically illuminated

#   --- output ---
    #   x*2 


def T_hapke(i, e, ph, Ref, w, g, t, h, B0):


    #   ラジアンに変換
    i_R = i * np.pi/ 180
    e_R = e * np.pi/ 180
    ph_R = ph * np.pi/ 180

    
    #	ph_R: phase angle in radian
	#	g: asymmetry parameter
    #   P: Phase nfunction

    """
    値の確認のための出力
    print("g="+str(g))
    print("PH="+str(ph_R))
    print("PHASE"+str(ph_R))
    print("cos(PHASE)="+str(np.cos(ph_R)))
    """

    P = (1 - g**2) / (1 + 2 * g * np.cos(ph_R) + g**2)**(3/2)

    
    #   theta_R: theta in radian 
    theta_R = t * np.pi / 180.
    xidz = 1. / sp.sqrt(1. + np.pi * (sp.tan(theta_R) ** 2))

    MUP, MU, S = roughness(i_R, e_R, ph_R, xidz, theta_R)
    
    M = (1 - sp.sqrt(1- w))/(1 + sp.sqrt(1 - w))
    B = B0 / ( 1 + np.tan(ph_R/2) /h)
    gamma = sp.sqrt(1 - w)
    H0 = HH(MUP, w)
    H = HH(MU, w)
    
    """
    print("P="+str(P))
    print("B="+str(B))
    print("H0="+str(H0))
    print("H="+str(H))
    print("S="+str(S))
    print("MUP="+str(MUP))
    print("MU="+str(MU))
    """

    diff = (abs((w/4*MUP/(MUP + MU)*((1 + B)*P + H0*H - 1)*S) - Ref))**2
    return diff


def HH(x,w):
    #	subroutine to calculate the multiple scatters
	#	--- Given ---
	#	x: angle in radian
	#	w: single scattering albedo

    r0 = (1 - sp.sqrt(1- w))/(1 + sp.sqrt(1 - w))
    return  1/(1-w*x*(r0+(1/2)*(1-2*r0*x)*np.log(float((1+x)/x))))

def roughness(i,e,ph,xidz,theta_R):
    cose = np.cos(e)
    sine = np.sin(e)
    cosi = np.cos(i)
    sini = np.sin(i)

    if ph == 180: # tan infinity
        f = sp.Integer(0)
    else:
        f = np.exp( -2 * np.tan(ph/2))
    
    if theta_R == 0 or i == 0:
        E1i = 0
        E2i = 0
    else:
        E1i = sp.exp(-2/np.pi/np.tan(theta_R)/sp.tan(i))
        E2i = sp.exp(-1/np.pi/np.tan(theta_R)**2/sp.tan(i)**2)
        #print ("E1i="+str(E1i))
        #print ("E2i="+str(E2i))
        
    if theta_R == 0 or e == 0:
        E1e = 0
        E2e = 0
    else:
        E1e = sp.exp(-2/np.pi/np.tan(theta_R)/np.tan(e))
        E2e = sp.exp(-1/np.pi/np.tan(theta_R)**2/np.tan(e)**2)
        #print ("E1e="+str(E1e))
        #print ("E2e="+str(E2e))

    #各式の下に書いてあるコメントはperlで書かれたもの
    if i <= e:
        #print(cosi)
        mu0e = xidz * (cosi + sini * sp.tan(theta_R) * ((np.cos(ph) * E2e + (np.sin(ph / 2) ** 2) * E2i )/ (2 - E1e - (ph/ np.pi) * E1i)))
        #mu0e = $chi*(cos($i) + sin($i)*tan($t)*(cos($p)*$E2e + sin(0.5*$p)**2*$E2i)/(2 - $E1e - $p/pi*$E1i));
        mue = xidz * (cose + sine * sp.tan(theta_R) * (E2e - (np.sin(ph / 2) ** 2) * E2i) / (2 - E1e - (ph / np.pi) * E1i))
        #mue  = $chi*(cos($e) + sin($e)*tan($t)*($E2e - sin(0.5*$p)**2*$E2i)/(2 - $E1e - $p/pi*$E1i));

        mu0e_0 = xidz * (cosi + sini * sp.tan(theta_R) * E2e / (2 - E1e))
        mue_0 = xidz * (cose + sine * sp.tan(theta_R) * E2e / (2 - E1e))
        
        S = mue/mue_0 * np.cos(i)/mu0e_0 * xidz/(1 - f + f*xidz*(np.cos(i)/mu0e_0))
        #$S = $mue/$mue_0 * $mu0/$mu0e_0 * $chi/(1 - $f + $f*$chi*($mu0/$mu0e_0));
    else:
        #print(cosi)
        mu0e = xidz * (cosi + sini * sp.tan(theta_R) * (E2i - np.sin(ph / 2) ** 2 * E2e) / (2 - E1i - (ph / np.pi) * E1e))
        #$mu0e = $chi*(cos($i) + sin($i)*tan($t)*($E2i - sin(0.5*$p)**2*$E2e)/(2 - $E1i - $p/pi*$E1e));
        mue = xidz * (cose + sine * sp.tan(theta_R) * (np.cos(ph) * E2i + np.sin(ph/ 2) ** 2 * E2e )/ (2 - E1i - (ph / np.pi) * E1e))
        #$mue  = $chi*(cos($e) + sin($e)*tan($t)*(cos($p)*$E2i + sin(0.5*$p)**2*$E2e)/(2 - $E1i - $p/pi*$E1e));
        
        mu0e_0 = xidz * (cosi + sini * sp.tan(theta_R) * E2i / (2 - E1i))
        mue_0 = xidz * (cose + sine * sp.tan(theta_R) * E2i / (2 - E1i))      

        S = mue/mue_0 * np.cos(i)/mu0e_0 * xidz/(1 - f + f*xidz*(np.cos(e)/mue_0))

    #print("mu0e="+str(mu0e))
    #print("mue="+ str(mue))
    #print("S="+str(S))

    return mu0e,mue, S,

def para(w, g, t, h, B0):

    #設定されたパラメータセットを確認するための出力

    # csvファイルから三つの角度とその時の実測の反射率を取り出す
    with open("/Users/satomichael/Downloads/dataset/comP1_data.csv") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        diff= 0
        cnt = 0

        for row in reader:
            diff  += T_hapke(float(row[0]),float(row[1]),float(row[2]),float(row[3]), w, g, t, h, B0)
            cnt = cnt +1
        print(w, g, t, h, B0, diff)

    return diff



# ---------Nelder-Mead---------------

# n次元の初期値を1個与えて、(n+1)　\times　n 次元の初期値を用意する
def get_x0(init_x, step=0.1):
    '''
    関数の次元をnとして初期値はn+1個用意する。
    初期値はまず1点(n次元)を指定し、残りのn個は各次元に一定の数字stepを足したものを使うのが慣例。
    :param init_x: ユーザが指定する1個の初期値
    :return: x の初期値 (n+1) 個
    '''
    dim = len(init_x)
    x0 = [init_x]
    for i in range(dim):
        x = copy.copy(init_x)
        x[i] += step
        x0.append(x)
    return np.array(x0)  # shape = (n+1,n)

def get_centroid(set_x_except_max_point):
    '''
    重心の計算
    :param set_x_except_max_point: 最大値をとる点を除いた点の座標
    :return: 最大値をとる点を除いた点の集合の重心
    '''
    _set_x_except_max_point = np.array(set_x_except_max_point)
    return np.sum(_set_x_except_max_point[:,0]/len(set_x_except_max_point)) # [[x_1,y_1],...,[x_n,y_n]]なのでx_iの平均を求める

def get_reflection_point(worst_point, centroid, alpha=1.0):
    '''反射点の計算
    :param worst_point: 最大値となる点
    :param centroid: 最大値となる点を除いた重心
    :param alpha: 反射点の係数
    :return: 反射点
    '''
    return centroid + alpha*(centroid - worst_point)

def get_expansion_point(worst_point, centroid, beta=2.0):
    '''
    反射拡大点の計算
    :param worst_point: 最大値をとる点
    :param centroid: 最大値となる点を除いた重心
    :param beta: 反射拡大点の係数
    :return: 反射拡大点
    '''
    return centroid + beta*(centroid - worst_point)

def get_outside_contraction_point(worst_point, centroid, gamma=0.5):
    '''反射縮小点の計算
    :param worst_point: 最大値をとる点
    :param centroid: 最大値をとる点を除いた重心
    :param gamma: 反射縮小点の係数
    :return: 反射縮小点'''
    return centroid + gamma * (centroid - worst_point)

def get_inside_contraction_point(worst_point, centroid, gamma=0.5):
    '''
    収縮点の計算
    :param worst_point: 最大値をとる点
    :param centroid: 最大値となる点を除いた重心
    :param gamma: 縮小点の係数
    :return: 縮小点のx
    '''
    return centroid - gamma * (centroid - worst_point)

def get_shrinkage_point(set_x_y, best_point, delta=0.5):
    '''
    縮小点の計算
    :param set_x_y: 点集合XとY
    :param best_point: 最小値をとる点
    :param delta: 最小点の係数
    :return: 最小点
    '''
    return [best_point + delta * (x - best_point) for x, y in set_x_y]

def core_algorithm_in_nelder_mead(func, set_x_y):
    '''
    ネルダーミード法のアルゴリズムの核
    :param func: 最小値を探索したい関数
    :param set_x_y: 点集合 X と Y [[x1,y1],[x2,y2],...,[xN,yN]]
    '''
    
    # step1: 最小、最大、２番目に最大を計算
    best_point = set_x_y[0][0]
    best_score = set_x_y[0][1]
    
    worst_point = set_x_y[-1][0]
    worst_score = set_x_y[-1][1]
    
    second_worst_point = set_x_y[-2][0]
    second_worst_score = set_x_y[-2][1]
    
    # step2: 最大値をとる点を除いたcentroidの計算
    centroid = get_centroid(set_x_y[:-1])
    
    # step3: 反射点の計算
    reflection_point = get_reflection_point(worst_point, centroid)
    reflection_score = func(reflection_point)
    
    # step4: 反射点と最大点の入れ替え
    if best_score <= reflection_score < second_worst_score :
        del set_x_y[-1]
        set_x_y.append([reflection_point, reflection_score])
        return set_x_y
    elif reflection_score < best_score:
        # step5: 反射拡張点もしくは反射点と最大点の入れ替え
        expansion_point = get_expansion_point(worst_point,centroid)
        expansion_score = func(expansion_point)
        if expansion_score < reflection_score:
            del set_x_y[-1]
            set_x_y.append([expansion_point, expansion_score])
            return set_x_y
        else:
            del set_x_y[-1]
            set_x_y.append([reflection_point, reflection_score])
            return set_x_y
    elif second_worst_score <= reflection_score:
        # step7: 反射収縮点の計算
        outside_contraction_point = get_outside_contraction_point(worst_point,centroid)
        outside_contraction_score = func(outside_contraction_point)
        # step8: 反射収縮点と最大点の入れ替え
        if outside_contraction_score < worst_score:
            del set_x_y[-1]
            set_x_y.append([outside_contraction_point, outside_contraction_score])
            return set_x_y
    # step9: 縮小点を計算して新たに集合Xと集合Yを算出
    shrinkage_point_list = get_shrinkage_point(set_x_y, best_point)
    shrinkage_score_list = [func(reduction_point) for reduction_point in shrinkage_point_list]
    reduction_value = zip(shrinkage_point_list, shrinkage_score_list)
    return reduction_value

def get_solution_by_nelder_mead(func, init_x, no_improve_thr=10e-8, no_improv_break=10, max_iter=0):
    '''
    ネルダーミード法の実行
    :param func: 最小値を探索したい関数
    :param init_x: 探索を開始する点
    :param no_improve_thr: 許容誤差
    :param no_improve_break: 一定階数最小値が更新されなかったら探索をやめる
    :param max_iter: 最大試行階数
    :return: 推定最小値と最適解
    '''
    
    set_x = get_x0(init_x)  # 集合 X の初期値を取得
    set_y = [func(x) for x in set_x] # 集合 Y の初期値を計算
    
    # step1: 集合XとYをYをキーにして昇順にソート
    set_x_y = zip(set_x,set_y)
    set_x_y = sorted(set_x_y, key=lambda t: t[1])
    
    prev_best_score = set_x_y[0][1]
    
    is_not_saturate = True  # 本に誤植あり (sarturate -> saturate)    
    no_improv = 0
    iters = 0
    while is_not_saturate:
        best_value = set_x_y[0]
        best_score = set_x_y[0][1]
        
        if max_iter and iters >= max_iter:
            print('iters : ', iters)
            return best_value
        if best_score < prev_best_score - no_improve_thr: # 改善度が敷居値を越えれば
            no_improv = 0
            prev_best_score = best_score
        else:
            no_improv += 1
        if no_improv >= no_improv_break:
            print('iters : ', iters)
            return best_value
        prev_best_score = set_x_y[0][1]
        set_x_y = core_algorithm_in_nelder_mead(func, set_x_y)
        set_x_y = sorted(set_x_y,key=lambda t :t[1])
        iters += 1
    
def rosen(x):
    return para(x[0],x[1],x[2],x[3],x[4])

# nelder mead法の実行
def main():
    x0 = np.array([0.06, 0.33, 32.8, 0.19, 0.91])

    res = get_solution_by_nelder_mead(rosen, x0)
    print('estimated min: ', res[1])
    print('x: ',res[0])

# KeyboardInterrupt

if __name__ == "__main__":
    main()

