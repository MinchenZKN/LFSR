clear all
clc
% load files
file_path = 'D:\Python\Super resolution\SR v2\networks\SR_3D_v9\';
t_history = readtable(strcat(file_path, 'training_history.csv'),'Delimiter',',');
v_history = readtable(strcat(file_path, 'validation_history.csv'),'Delimiter',',');
num_samples_t = size(t_history,1);
num_samples_v = size(v_history,1);

t_record = table2array(t_history(:,7));
windowSize = floor(size(t_record,1)/10); 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
data_f = filter(b,a,t_record);
min_record = ones(size(t_record,1),1).* min(t_record);
curr_trend = ones(size(t_record,1),1).* data_f(end,1);
min_trend = ones(size(t_record,1),1).* min(data_f);
s1 = strcat('min of Loss = ',num2str(min(t_record)));
s2 = strcat('current trend level training = ',num2str(data_f(end,1)));
if 0 == min(data_f) - data_f(end,1)
    S = 'Yes';
else
    S = 'No';
end
s5 = strcat('Recent Trend Descent = ',S);
s6 = strcat('Recent Trend Increase = ',num2str(-(min(data_f) - data_f(end,1))));


v_record = table2array(v_history(:,7));
data_ff = filter(b,a,v_record);
min_record_v = ones(size(v_record,1),1).*min(v_record);
curr_trend_v = ones(size(v_record,1),1).*data_ff(end,1);
min_trend_v = ones(size(v_record,1),1).*min(data_ff(:,1));
s3 = strcat('min of Loss = ',num2str(min(v_record)));
s4 = strcat('current trend level validation = ',num2str(data_ff(end,1)));


figure('name','LF cvSR Loss Recent Overview');
subplot(2,1,1),
ylim([min(data_ff)/2, max(data_ff(end-1000:end))]);
semilogy(data_ff(end-100:end),'y', 'LineWidth', 2)
hold on
semilogy(curr_trend_v(end-100:end), 'g', 'LineWidth', 1)
legend('Loss Trend Validation',s4,'Location','best');
title('Loss Trend')
subplot(2,1,2),
ylim([min(data_f)/2, max(data_f(end-1000:end))]);
semilogy(data_f(end-100:end), 'r', 'LineWidth', 2)
hold on
semilogy(curr_trend(end-100:end), 'k', 'LineWidth', 1)
legend('Loss Trend Training',s2,'Location','best');
title(s6)

spot = 500;
data_f(1:spot,1) = NaN;
data_ff(1:spot,1) = NaN;


figure('name','LF cvSR Loss Overview');
semilogy(t_record, 'b')
hold on
semilogy(data_f, 'r', 'LineWidth', 2)
hold on
semilogy(data_ff, 'y', 'LineWidth', 2)
hold on
semilogy(min_record, 'b', 'LineWidth', 4)
hold on
semilogy(curr_trend, 'k', 'LineWidth', 1)
hold on
semilogy(curr_trend_v, 'g', 'LineWidth', 1)
lgd = legend('Loss Value Training ','Loss Trend Training','Loss Trend Validation',s1,s2,s4,'Location','best');
title(lgd,'Plot Details')
title(s5)


figure;
semilogy(v_record, 'b')
hold on
semilogy(data_ff, 'r', 'LineWidth', 2)
hold on
semilogy(min_record_v, 'g', 'LineWidth', 2)
hold on
semilogy(min_trend_v, 'k', 'LineWidth', 1)
legend('Loss value Validation ','Loss trend Validation',s3,s4)
title('cv SR validation loss')




% params
% start_col = 6;
% 
% name_col = start_col:2:size(t_history,2);
% val_col = start_col+1:2:size(t_history,2);
% 
% % reshuffle t_history
% names = table2array(t_history(1,start_col:2:end));
% t_history_new = t_history;
% 
% for i = 1:length(names)
%     data_new = NaN(num_samples_t,1);
%     name_new = cell(num_samples_t,1);
%     for j = 1:length(name_col)
%         data_name = table2array(t_history(:,name_col(j)));
%         data_value = table2array(t_history(:,val_col(j)));
%         index = cellfun(@(x) strcmp(x,names{i}), data_name, 'UniformOutput', 1);
%         data_new(index)= data_value(index);
%         name_new(index) = data_name(index);
%         clear data_name data_value
%     end
%     t_history_new(:,name_col(i)) = name_new;
%     t_history_new(:,val_col(i)) = array2table(data_new);
%     clear name_new data_new
% end
% 
% t_history = t_history_new;
% 
% % reshuffle v_history
% names = table2array(v_history(1,start_col:2:end));
% v_history_new = v_history;
% 
% for i = 1:length(names)
%     data_new = NaN(num_samples_v,1);
%     name_new = cell(num_samples_v,1);
%     for j = 1:length(name_col)
%         data_name = table2array(v_history(:,name_col(j)));
%         data_value = table2array(v_history(:,val_col(j)));
%         index = cellfun(@(x) strcmp(x,names{i}), data_name, 'UniformOutput', 1);
%         data_new(index)= data_value(index);
%         name_new(index) = data_name(index);
%         clear data_name data_value
%     end
%     v_history_new(:,name_col(i)) = name_new;
%     v_history_new(:,val_col(i)) = array2table(data_new);
%     clear name_new data_new
% end
% 
% v_history = v_history_new;
% 
% clear t_history_new v_history_new
% 
% % filter
% windowSize = floor(size(t_history,1)/10); 
% b = (1/windowSize)*ones(1,windowSize);
% a = 1;
% 
% % training
% for i = 1:length(name_col)
%     data = table2array(t_history(:,val_col(i)));
%     data(isnan(data) == 1) = [];
%     data_f = filter(b,a,data);
%     name = table2array(t_history(:,name_col(i)));
%     name = name{1};
%     figure; semilogy(data, 'b')
%     hold on
%     semilogy(data_f, 'r', 'LineWidth', 2)
%     title(strcat(name,32,'training'))
%     clear data
% end
% 
% % validation
% for i = 1:length(name_col)
%     data = table2array(v_history(:,val_col(i)));
%     data(isnan(data) == 1) = [];
%     data_f = filter(b,a,data);
%     name = table2array(v_history(:,name_col(i)));
%     name = name{1};
%     figure; semilogy(data)
%     hold on
%     semilogy(data_f, 'r', 'LineWidth', 2)
%     title(strcat(name,32,'validation'))
%     clear data
% end
% 
