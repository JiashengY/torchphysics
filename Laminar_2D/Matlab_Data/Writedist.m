load('/net/istmhome/users/hi224/Dokumente/simson_Postprocessing/ML_ks/Round_0/2289/01/xy_tot_T.mat')
titles=["x","dist"];
dist_1d=dist(:,1);
datamatrix=[linspace(0,max(x),length(dist_1d))',dist_1d];
data=[titles;datamatrix];
%csvwrite("Flat_Turb_500.csv",data)
writematrix(data,'DIST_2289.csv','Delimiter',',')