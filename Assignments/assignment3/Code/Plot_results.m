%   function [ ] = Plot_results(t_loss,t_cost,t_acc,v_loss,v_cost,v_acc,yline)
%     

function [ ] = Plot_results(t_loss,t_cost,t_acc,v_loss,v_cost,v_acc,yline,etas)
 figure()
a1 = plot(yline,t_cost,'g-');
M1 = 'Training cost ';
hold on
a2 = plot(yline,v_cost,'b-');
M2 = 'Validation cost ';
legend(M1,M2);
title(" Cost each update step");
xlabel(" update step  ");
ylabel(" Cost Function ");
% ylim([1.5,3]);
hold off

figure()
a1 = plot(yline,t_acc,'g-');
M1 = 'Training Accuracy ';
hold on
a2 = plot(yline,v_acc,'b-');
M2 = 'Validation Accuracy ';
legend(M1,M2);
title(" Accuracy each update step");
xlabel(" update step  ");
ylabel(" Accuracy ");
hold off

figure()
a1 = plot(etas,'g-');
M1 = 'etas ';
legend(M1);
title(" etas");
xlabel(" update step");
ylabel("etas ");
hold off

figure()
a1 = plot(yline,t_cost,'g-');
M1 = 'Training loss ';
hold on
a2 = plot(yline,v_cost,'b-');
M2 = 'Validation loss ';
legend(M1,M2);
title(" Loss each update step");
xlabel(" update step  ");
ylabel(" Loss Function ");
% ylim([1.5,3]);
hold off
end