load('xy_tot_T.mat')
titles=["x","y","U","V","urms","vrms","uv"];
datamatrix=[zeros(length(y),1),y,Umean',Vmean',urms',vrms',uv'];
data=[titles;datamatrix];
%csvwrite("Flat_Turb_500.csv",data)
writematrix(data,'Flat_Turb_500.csv','Delimiter',',')