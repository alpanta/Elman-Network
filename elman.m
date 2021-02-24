y(1)=0.1;   % Initial conditions 
y(2)=0.1;

index=1;% Variable for plotting test data
sw=2; % Variable to switch between random and nearly best initial 
      % conditions for weights (sw=1 for random, sw=2 for nearly best)

nnu = 1;   % Input layer
nnx = 6;   % Hidden layer
nny = 1;   % Output layer

training_set_size=900;      % Training data(First 900 elements of 
                            % Billings System)
test_set_size =training_set_size+40;  % Test data

iteration=1;
momentum=0.65;

if sw==1
    n=0.029; % Learning rate
    weights_u = randn(nnx,nnu);  % Initial conditions for weights
    weights_x = randn(nnx,nnx);
    weights_y = randn(nny,nnx);
end

if sw==2
     n=0.027; % Learning rate 

     % Nearly best initial conditions for weights (Recorded weights)
     weights_x = [

        0.8908    1.8106    1.6085   -1.2210   -1.0242    0.5656;
       -1.8913    0.5387   -0.5894   -0.0360    0.4159   -1.8232;
        1.1209    1.0958   -1.2613    0.5687   -2.3102   -3.7303;
       -0.3719   -0.8632   -0.4843   -0.6312   -0.2901    1.6408;
       -1.5000   -1.7704   -0.1812    0.9253   -0.6058    1.5762;
        0.8455    1.8453    0.9314    0.9002   -1.9390    0.2969];

     weights_u =[

       -0.8333;
        0.2984;
       -0.4605;
        0.4958;
       -1.3770;
        0.0835];

    weights_y =[

        0.1797    0.0751    0.1256    0.0727   -0.3476    0.3671];

end


wxold = zeros(nnx,nnx); % Weights to hold previous weight 
wuold = zeros(nnx,nnu); % values for calculating momentum term
wyold = zeros(nny,nnx);

xold = zeros(nnx,1); % Previous input values 

for i=1:iteration
    % Training 
    for k = 3:training_set_size

        input = normrnd(0,0.01);     % Noise implemented for input 
        % Billings System
        y(k) = (0.8 - 0.5 * exp(-y(k-1)^2))*y(k-1)...
                - (0.3 + 0.9*exp(-y(k-1)^2))*y(k-2)...
                + 0.1*sin(pi*y(k-1)) + input;


        % Feedforward 
        v = weights_u * [input]  + weights_x * xold;

        [x,f_der] = acti_func(v);

        xold = x;

        y_network(k) = weights_y * x;

        e = y(k) - y_network(k);    % Error

        % Trivial variables for momentum term calculation
        tempx = weights_x;
        tempu = weights_u;
        tempy = weights_y;

        % Updates of weights 
        weights_x = weights_x + n *( weights_y'*e).*f_der*x'... 
                    + momentum*(weights_x - wxold);
        weights_u = weights_u + n *( weights_y'*e).*f_der*[input]'...
                    + momentum*(weights_u - wuold);
        weights_y = weights_y + n*e*x' + momentum*(weights_y - wyold);

        wxold = tempx;
        wuold = tempu;
        wyold = tempy;


    end


end

% Test
for k = training_set_size:test_set_size
input = normrnd(0,0.01);
y(k) = (0.8 - 0.5 * exp(-y(k-1)^2))*y(k-1)... 
        - (0.3 + 0.9*exp(-y(k-1)^2))*y(k-2)...
        + 0.1*sin(pi*y(k-1)) + normrnd(0,0.01);

    y1(index)=y(k);

    v = weights_u * [input] + weights_x * xold;

    [x,f_der] = acti_func(v);

    xold = x;

    y_network(k) = weights_y * x;

    y2(index)=y_network(k);

    index=index+1;
end

figure(1)  % First 150 elements of Billings System
plot(y)
title("Billings System");
axis([0 150 -1.5 1.5])

figure(2)
plot(y1,'-*r'); hold on
plot(y2,'-ob'); hold on
title("System Outputs vs. Network Outputs");
legend('System Outputs','Network Outputs')


function [a,der] = acti_func(value)  % Activation function

             a   =  tanh(value);          
             der =  1 - tanh(value).^2;   % Derivative of activation func.
end
