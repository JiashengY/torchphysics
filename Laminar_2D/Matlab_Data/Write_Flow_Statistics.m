load('/net/istmhome/users/hi224/Dokumente/simson_Postprocessing/ML_ks/Round_0/1113/01/xy_tot_T.mat')
titles=["x","y","U","V","W","urms","vrms","wrms","uv","uw","vw"];
datamatrix=[zeros(length(y),1),y,Umean',Vmean',Wmean',urms',vrms',wrms',uv',uw_mean',vw_mean'];
data=[titles;datamatrix];
%csvwrite("Flat_Turb_500.csv",data)
writematrix(data,'irr_turb_1113.csv','Delimiter',',')