exp_corr_data = importdata('data_for_matlab_corr.csv');
line_loc = [41 83 124 164 205 241 282 324 366 407 447 488 517 560 601];
cs = cbrewer('div','Spectral',128);
cs(cs>1) = 1;
cs(cs<0) = 0;
figure;
imagesc(exp_corr_data.data);
colormap(flipud(cs))
caxis([-0.3 0.3])
colorbar
hold on
for i = 1:length(line_loc)
    plot(xlim,[line_loc(i) line_loc(i)],'k')
    plot([line_loc(i) line_loc(i)],ylim,'k')
end
axis square
set(gcf,color,'w')