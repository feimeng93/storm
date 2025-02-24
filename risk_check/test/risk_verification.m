function status = risk_verification(ec,interval)
    % check for three ellipses of a 7dof kinova
    % ec : ellipse centers positions for 3 ellipses (middl) over 11 time steps, size: 11x3x3, links: 1&2,3&4,5&6 

    % 2s
    tic
    n_ellipses = 3;
    t0 = interval(1);
    tf = interval(end);
    assert(length(interval) == size(ec, 1), 'interval should align with ec first dimension');
        % Risk Contours Delta=0.01, Being Safe: >=0 
    % C * A_SOS_Cons_1>=0 (C is a alrge positive number to avoid numerical issues) and A_SOS_Cons_2>=0
    Safe_1 = @(x1,x2,x3,t) (58146*x1)/78125 + (58146*x2)/78125 + (58146*x3)/78125 - (99*x1^2*x2^2)/50 - (99*x1^2*x3^2)/50 - (99*x2^2*x3^2)/50 - (792*x1*x2)/625 - (792*x1*x3)/625 - (792*x2*x3)/625 + (198*x1*x2^2)/125 + (198*x1^2*x2)/125 + (198*x1*x3^2)/125 + (198*x1^2*x3)/125 + (198*x2*x3^2)/125 + (198*x2^2*x3)/125 + (x1^2 - (4*x1)/5 + x2^2 - (4*x2)/5 + x3^2 - (4*x3)/5 + 881/1875)^2 - (48873*x1^2)/31250 + (198*x1^3)/125 - (48873*x2^2)/31250 - (99*x1^4)/100 + (198*x2^3)/125 - (48873*x3^2)/31250 - (99*x2^4)/100 + (198*x3^3)/125 - (99*x3^4)/100 - 25199662621378194455071203/115292150460684697600000000;
    Safe_2 = @(x1,x2,x3,t) x1^2 - (4*x1)/5 + x2^2 - (4*x2)/5 + x3^2 - (4*x3)/5 + 881/1875;

    % link size with extended the next link
    % ls = [0.0921046, 0.0949995, 0.001, 0.1026092, 0.001, 0.0733221, 0.001 ;  % size in x
    % 0.0921046, 0.0949995, 0.001, 0.1026092, 0.001, 0.0733221, 0.001 ; % size in y
    % 0.172, 0.512755, 0.001, 0.39543, 0.001, 0.205439, 0.001] ;  % size in x
    % base link size: [0.092,0.092,0.17085]

    % ellipse_size
    es = [0.0949995, 0.1026092, 0.0733221 ;  % size in x
    0.0949995, 0.1026092, 0.073322 ; % size in y
    0.512755, 0.39543, 0.205439] ;  % size in x
    t = sdpvar(1,1);

    px = zeros(n_ellipses, 4);
    py = zeros(n_ellipses, 4);
    pz = zeros(n_ellipses, 4);
    Px = cell(n_ellipses,1);
    Py = cell(n_ellipses,1);
    Pz = cell(n_ellipses,1);

    for i = 1:n_ellipses
        px(i,:) = polyfit(interval, ec(:,i,1), 3);
        py(i,:) = polyfit(interval, ec(:,i,2), 3);
        pz(i,:) = polyfit(interval, ec(:,i,3), 3);
        Px{i} = px(i,1)*t.^3 + px(i,2)*t.^2 + px(i,3)*t + px(i,4);
        Py{i} = py(i,1)*t.^3 + py(i,2)*t.^2 + py(i,3)*t + py(i,4);
        Pz{i} = pz(i,1)*t.^3 + pz(i,2)*t.^2 + pz(i,3)*t + pz(i,4);
    end

    status_1 = func_3D_SOS_Tube_yalmip(Safe_1, t0, tf, Px, Py, Pz, es, ec,n_ellipses);
    status_2 = func_3D_SOS_Tube_yalmip(Safe_2, t0, tf, Px, Py, Pz, es, ec,n_ellipses);
    status = (status_1 == 1 && status_2 == 1);
    disp(['risk verification time: ',num2str(toc)]);
end

function status=func_3D_SOS_Tube_yalmip(Safe,t0,tf,Px,Py,Pz,ls,lc, N)  
    % Safe: Safety Constraints, Obs(x) <=0 ---> Safe(x)>=0 --->Safe(x(t),y(t),z(t))>=0 for all t0 =<t=< tf and (xt,yt) in {R^2-xt^2-yt^2-z^2} 
    % (Px,Py,Pz): given polynomial trajectory
    % [t0,tf]:start and final time, i.e., t in [t0 tf]
    % R: radius of tube
    % d: SOS relaxation order d>= order of given polynomial trajectory
    % ls: link size
    % lc: link center

    % % Initialize parallel pool if not already running
    % if isempty(gcp('nocreate'))
    %     parpool('local');
    % end
    % Pre-allocate arrays
    F = cell(N,1);
    C = cell(N,1);

    % Parallel loop
    for i = 1:N
        sdpvar t xt yt zt
        % Process each ellipse in parallel
        [s1,c1] = polynomial(t,2);
        [s2,c2] = polynomial(t,2);
        
        % Build constraints for this ellipse
        F{i} = [sos(Safe(Px{i}+xt,Py{i}+yt,Pz{i}+zt,t)-s1*(t-t0)*(tf-t)-s2*(1-(xt-lc(1,i,1))^2/ls(1,i)^2-(yt-lc(1,i,2))^2/ls(2,i)^2-(zt-lc(1,i,3))^2/ls(3,i)^2)), sos(s1), sos(s2)];
        % F{i} = [sos(Safe(Px{1}+xt,Py{1}+yt,Pz{1}+zt,t)-s1*(t-t0)*(tf-t)-s2*(1-(xt-lc(1,1,1))^2/ls(1,1)^2-(yt-lc(1,1,2))^2/ls(2,1)^2-(zt-lc(1,1,3))^2/ls(3,1)^2)), sos(s1), sos(s2)];
        C{i} = [c1,c2];
    end

    % Combine results
    F_combined = [];
    C_combined = [];
    for i = 1:N
        F_combined = [F_combined, F{i}];
        C_combined = [C_combined, C{i}];
    end
    
    ops = sdpsettings('solver','mosek','verbose', 0, 'mosek.MSK_IPAR_LOG', 0);
    sol = solvesos(F_combined,[],ops,C_combined);
    
    status = (sol.problem == 0);
end


