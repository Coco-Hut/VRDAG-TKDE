import numpy as np
import pandas as pd
import scipy.stats as ss
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from networkx.algorithms.centrality import betweenness_centrality,closeness_centrality
from metrics_utils import *
import networkx as nx
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def get_mean_var_std(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr,ddof=1)
    return arr_mean ,arr_std

def hellinger_distance(p, q):
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    
    distance = np.sqrt(np.sum((sqrt_p - sqrt_q) ** 2)) / np.sqrt(2)
    
    return distance

def JS_divergence(p,q):
    M = (p+q)/2
    return 0.5 * ss.entropy(p,M,base=2) + 0.5*ss.entropy(q,M,base=2)

class StrucEvaluator:
    
    def __init__(self,eval_time_len,eval_method='med',
                 verbose=False,is_ratio=False,mmd_beta=1.0) -> None:
        
        '''
        eval_time_len: evaluated time length
        eval_method: evaluation method
        '''
        
        self.eval_time_len=eval_time_len
        self.eval_method=eval_method
        self.verbose=verbose 
        self.is_ratio=is_ratio 
        self.mmd_beta=mmd_beta 
        self.stats_map={'deg_dist_in':self.stats_deg_dist,
                        'deg_dist_out':self.stats_deg_dist,
                        'clus_dist':self.stats_clus_dist,
                       'wedge_count':self.stats_wedge_count,
                       'n_components':self.stats_connected_components,
                       'lcc_size':self.stats_largest_connected_components,
                       'power_law_in':self.stats_power_law_exp,
                       'power_law_out':self.stats_power_law_exp,
                       }
        
        '''
        self.stats_map={'mean_degree':self.stats_mean_degree,
                       'wedge_count':self.stats_wedge_count,
                       'n_components':self.stats_connected_components,
                       'lcc_size':self.stats_largest_connected_components,
                       'power_law_in':self.stats_power_law_exp,
                       'power_law_out':self.stats_power_law_exp,
                       'node_div':self.stats_node_div_dist
                       }
        '''
    
    def error_func(self,org_graph,generated_graph,method='med'):
        
        if self.is_ratio==True:
            metric = np.divide(np.abs(org_graph - generated_graph), np.abs(org_graph))
        else:
            metric = np.abs(org_graph - generated_graph)

        if method=='med':
            return np.median(metric)
        elif method=='mean':
            return np.mean(metric)
        else:
            raise ValueError('{} method does not exist!'.format(method))
    
    def error_func_dist_mmd(self,org_dist,gen_dist,method='med'):
        
        metric=np.array([calculate_mmd(org_dist[t].reshape(-1,1),gen_dist[t].reshape(-1,1),beta=self.mmd_beta) for t in range(self.eval_time_len)])
        
        if method=='med':
            return np.median(metric)
        elif method=='mean':
            return np.mean(metric)
        else:
            raise ValueError('{} method does not exist!'.format(method))    
    
    # 1. Mean Degree
    def stats_mean_degree(self,A_in,time_len):

        mean_degree=np.array([np.mean(A_in[t].sum(axis=0)) for t in range(time_len)])
        
        return mean_degree
    
    # 2. degree distribution
    def stats_deg_dist(self,A_in,time_len,flow='in'):
    
        deg_hist=[]
            
        for t in range(time_len):
            
            if flow=='out':
                deg_dist=list(A_in[t].sum(axis=1))
            else:
                deg_dist=list(A_in[t].sum(axis=0))
            
            hist_deg, _ = np.histogram(
            deg_dist, bins=100, range=(0, 100), density=False)
            
            deg_hist.append(hist_deg)
        
        return np.array(deg_hist)
    
    def stats_clus_dist(self,A_in,time_len):
        
        coeffs_hists=[]
        
        for t in range(time_len):
            
            coeffs_dist=list(nx.clustering(nx.from_numpy_array(A_in[t])).values())
            
            hist_clus, _ = np.histogram(
            coeffs_dist, bins=100, range=(0.0, 1.0), density=False)
            
            coeffs_hists.append(hist_clus)
            
        return np.array(coeffs_hists)
            
    # 2. Wedge Count
    def stats_wedge_count(self,A_in,time_len):
        
        n_wedge=np.array([wedge_count(A_in[t]) for t in range(time_len)])
        
        return n_wedge
    
    # 3. Global Clustering Coefficients
    def stats_global_cluster_coef(self,A_in,time_len):
    
        n_claw=np.array([claw_count(A_in[t]) for t in range(time_len)])
        n_triangle=np.array([triangle_count(A_in[t]) for t in range(time_len)])
        
        global_cluster_coef=np.array([0 if n_claw[t]==0 else 3*n_triangle[t]/n_claw[t] for t in range(time_len)])
        
        return global_cluster_coef
    
    # 4. Number of connected components
    def stats_connected_components(self,A_in,time_len):
    
        cc=np.array([connected_components(A_in[t],directed=True)[0] for t in range(time_len)])
        
        return cc
    
    # 5. Largest Connected Components Size
    def stats_largest_connected_components(self,A_in,time_len):
    
        lcc_size=np.array([LCC_size(A_in[t]) for t in range(time_len)])
        
        return lcc_size
    
    # 6/7 Power Law Exponent - (in/out)
    def stats_power_law_exp(self,A_in,time_len,flow='out'):
    
        PLE=np.array([power_law_exp(A_in[t],flow) for t in range(time_len)])
        
        return PLE
    
    # 8. Node Divergence Distribution
    def stats_node_div_dist(self,A_in,time_len):
            
        node_div_hists=[]
        
        node_div=[node_div_dist(A_in[t]).reshape(-1) for t in range(time_len)]
        
        for t in range(time_len):
        
            hist_div, _ = np.histogram(
            node_div[t], bins=100, range=(-1.0, 1.0), density=False)
                
            node_div_hists.append(hist_div)
        
        return np.array(node_div_hists)
    
    # Note: BC and CC needs long running time! 
    
    def stats_gini(self,A_in,time_len,flow='out'):
        
        Gini=np.array([gini(A_in[t],flow) for t in range(time_len)])
        
        return Gini
    
    # 9. Mean Betweeness Centrality
    def stats_mean_betweeness_centrality(self,G,time_length): 

        mean_bc=np.array([np.mean(list(betweenness_centrality(G[t]).values())) for t in range(time_length)])
        
        return mean_bc
    
    # 10. Mean Closeness Centrality
    def stats_mean_closeness_centrality(self,G,time_length): 

        mean_bc=np.array([np.mean(list(closeness_centrality(G[t]).values())) for t in range(time_length)])
        
        return mean_bc
    
    # output the evaluation results for all statistics
    def comp_graph_stats(self,A_src,A_gen,stats_list=None):
        
        res_dict=defaultdict(float) # result dictionary
        
        metric_list=stats_list if stats_list is not None else list(self.stats_map.keys())
    
        for stat in metric_list:
            
            if stat=='power_law_in' or stat=='gini_in':
                res_dict[stat]=self.error_func(self.stats_map[stat](A_src,
                                                                self.eval_time_len,'in'),
                                                self.stats_map[stat](A_gen,
                                                                self.eval_time_len,'in'),self.eval_method)
            elif stat=='power_law_out'or stat=='gini_out':
                res_dict[stat]=self.error_func(self.stats_map[stat](A_src,
                                                                self.eval_time_len,'out'),
                                                self.stats_map[stat](A_gen,
                                                                self.eval_time_len,'out'),self.eval_method)
            
            elif stat == 'deg_dist_in':
                res_dict[stat]=self.error_func_dist_mmd(self.stats_map[stat](A_src,
                                                                self.eval_time_len,'in'),
                                                        self.stats_map[stat](A_gen,
                                                                self.eval_time_len,'in'),self.eval_method)
            
            elif stat == 'deg_dist_out':
                res_dict[stat]=self.error_func_dist_mmd(self.stats_map[stat](A_src,
                                                                self.eval_time_len,'out'),
                                                        self.stats_map[stat](A_gen,
                                                                self.eval_time_len,'out'),self.eval_method)
            
            elif stat in ['clus_dist','node_div']:
                res_dict[stat]=self.error_func_dist_mmd(self.stats_map[stat](A_src,
                                                                self.eval_time_len),
                                                        self.stats_map[stat](A_gen,
                                                                self.eval_time_len),self.eval_method)
            
            else:
                res_dict[stat]=self.error_func(self.stats_map[stat](A_src,
                                                                self.eval_time_len),
                                                self.stats_map[stat](A_gen,
                                                                self.eval_time_len),self.eval_method)

            if self.verbose:
                print('{} metirc finished!'.format(stat))
            
        return res_dict

'''
class AttrEvaluator:
    
    def __init__(self,eval_time_len,num_bins=50) -> None:
        
        self.eval_time_len=eval_time_len
        self.num_bins=num_bins 
    
    def degree_measure(self,A_src,A_gen,X_src,X_gen,n_box,deg_sep):

        attr_div_list=[] 
        ensemble_js_div=[] 
        
        bins=[i*deg_sep for i in range(n_box)]
        
        for t in range(self.eval_time_len):
            
            src_degree = A_src[t].sum(axis=0) + A_src[t].sum(axis=1)
            gen_degree= A_gen[t].sum(axis=0) + A_gen[t].sum(axis=1)

            box_src,w=self.box_cut(src_degree,n_box,bins)
            box_gen,_=self.box_cut(gen_degree,n_box,bins)
            
            all_attr_div=[] 
            ensemble_div=[] 
            
            for b in range(n_box):
                attrs_js_div,sum_js_div=self.JS_div_X(X_src[t][box_src[b]],X_gen[t][box_gen[b]])
                all_attr_div.append(attrs_js_div)
                ensemble_div.append(sum_js_div)
            
            weighted_attr_div=[w[b]*all_attr_div[b] for b in range(n_box)] # attr_dim x n_box
            attr_div_vec=weighted_attr_div[0]
            for b in range(1,n_box):
                attr_div_vec+=weighted_attr_div[b]

            attr_div_list.append(attr_div_vec)
            
            ensemble_js_div.append(sum([w[b]*ensemble_div[b] for b in range(n_box)]))
            
        return attr_div_list,np.median(ensemble_js_div)
            
    def box_cut(self,degrees,n_box,bins):
        
        w_t=np.array([0 for _ in range(n_box)]) # 权重与原分布箱子内的节点数乘正比
        box=[None for _ in range(n_box)] # 每个箱子存储
        
        for b in range(n_box):
            if b!=n_box-1:
                box[b]=np.where((degrees>=bins[b])&(degrees<bins[b+1]))[0]
            else:
                box[b]=np.where(degrees>=bins[b])[0]
            w_t[b]=len(box[b])
        w_t=w_t/len(degrees)
        
        return box,w_t
    
    # 每个属性的分布求JS Divergence然后叠加，因为是相互独立的
    def JS_divergence(self,p,q):
        M = (p+q)/2
        return 0.5 * ss.entropy(p,M,base=2) + 0.5*ss.entropy(q,M,base=2)

    def JS_div_x(self,x_src,x_gen,num_bins):
        
        if len(x_src)==0:
            return 0
        elif len(x_src)!=0 and len(x_gen)==0:
            return np.log(2) # JS散度的最大值
        else:
            _max_ = max(np.max(x_src),np.max(x_gen))
            _min_ = min(np.min(x_src),np.min(x_gen))
            
            bins = np.linspace(_min_-1e-4,_max_+1e-4,num = num_bins)
            pdf_src = pd.cut(x_src,bins).value_counts()/len(x_src) # 原数据分布概率密度函数
            pdf_gen = pd.cut(x_gen,bins).value_counts()/len(x_gen) # 生成数据分布概率密度函数
            
            return self.JS_divergence(pdf_src.values,pdf_gen.values)

    def JS_div_X(self,X_src,X_gen,num_bins=50):
        
        js_div=[] # 记录各个属性的JS散度
        
        for attr in range(X_src.shape[1]):
            x_src,x_gen=X_src[:,attr],X_gen[:,attr]
            js_div_value=self.JS_div_x(x_src,x_gen,num_bins)
            js_div.append(js_div_value)
            
        return np.array(js_div),np.sum(np.array(js_div))

'''

def evaluate_attr(src_attrs=None,
                  gen_attrs='normal',
                  diff='emd',
                  bins=10,
                  method='mean',
                  low_bound=0,
                  upper_bound=1,
                  n_step=None):

    attr_distance=[] 
    spearman_gt=[] 
    spearman_gc=[] 
    
    for t in range(n_step):
        
        temp_attr_distance=[] # 对每个属性分别计算分布差异
        
        x_t=np.array(src_attrs[t])
            
        if x_t.shape[1]==2:
            
            if gen_attrs=='normal':

                gen_x=[]

                for a_id in range(x_t.shape[1]):
                    
                    mean,std=get_mean_var_std(x_t[:,a_id]) # 估计属性的分布
                    _gen_x=np.random.normal(loc=mean,scale=std,size=(1,x_t.shape[0])) # 生成的属性列表
                    gen_x.append(_gen_x)
                
                gen_x=np.concatenate([gen_x[0],gen_x[1]]) # N x 2
            
            else:
                gen_x=gen_attrs[t]
            
            for a_id in range(x_t.shape[1]):
        
                gen_dist, _ = np.histogram(
                    gen_x, bins=bins, range=(low_bound, 1), density=False)

                src_dist,_ = np.histogram(
            x_t[:,a_id], bins=bins, range=(low_bound, 1), density=True)
                
                src_dist=src_dist/np.sum(src_dist)
                gen_dist=gen_dist/np.sum(gen_dist)
                
                if diff=='emd':
                    distance=wasserstein_distance(src_dist,gen_dist)
                elif diff=='js':
                    distance=JS_divergence(src_dist,gen_dist)
                elif diff=='hellinger':
                    distance=hellinger_distance(src_dist,gen_dist)
                else:
                    raise ValueError('Wrong distance function')

                temp_attr_distance.append(distance)

            attr_distance.append(np.mean(temp_attr_distance))

            spearman_gt.append(spearmanr(x_t[:,0],x_t[:,1])[0])
            spearman_gc.append(spearmanr(gen_x[:,0],gen_x[:,1])[0])
        
        if x_t.shape[1]==1:
            
            if gen_attrs=='normal':
                              
                mean,std=get_mean_var_std(x_t[:]) 
                gen_x=np.random.normal(loc=mean,scale=std,size=(1,x_t.shape[0])) 
            
            else:
                gen_x=gen_attrs[t]
                
            gen_dist, _ = np.histogram(
                    gen_x, bins=bins, range=(low_bound, upper_bound), density=False)

            src_dist,_ = np.histogram(
            x_t[:], bins=bins, range=(low_bound, upper_bound), density=False)
            
            src_dist=src_dist/np.sum(src_dist)
            gen_dist=gen_dist/np.sum(gen_dist)
            
            if diff=='emd':
                distance=wasserstein_distance(src_dist,gen_dist)
            elif diff=='js':
                distance=JS_divergence(src_dist,gen_dist)
            elif diff=='hellinger':
                distance=hellinger_distance(src_dist,gen_dist)
            else:
                raise ValueError('Wrong distance function')

            attr_distance.append(distance)
    
    if method=='mean':
        if spearman_gt!=[]:
            return np.mean(attr_distance),np.mean(np.abs(np.array(spearman_gt) - np.array(spearman_gc)))
        else:
            return np.mean(attr_distance), None   

