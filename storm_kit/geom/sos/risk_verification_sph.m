function status = risk_verification(ec,er,interval, horizon)
    % check for balls of the 7 link kinova
    % ec : sphere centers positions of the links over 50 time steps, size: 50xNx3, N=3(lp)*shperes, link_pair: 1&2,3&4,5&6 
    % er: sphere raduis
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
    
    ell_to_test = [];

    assert(size(ell_to_test) == (horizon*3,1), 'ell_to_test should have 3*horizon dimensions for three link ellispoids at each time step in the horizon');
 
    statuses = ones(3,1);
    t = sdpvar(1,1);

    px = zeros(3, 4);
    py = zeros(3, 4);
    pz = zeros(3, 4);
    for e_idx = 1:3
        px(e_idx,:) = polyfit(interval, ec(:,e_idx,1), 3);
        py(e_idx,:) = polyfit(interval, ec(:,e_idx,2), 3);
        pz(e_idx,:) = polyfit(interval, ec(:,e_idx,3), 3);
    end
    
    tic
    parfor i =1:horizon*3
    for e_idx = 1:3
        Px = px(e_idx,1)*t.^3 + px(e_idx,2)*t.^2 + px(e_idx,3)*t + px(e_idx,4);
        Py = py(e_idx,1)*t.^3 + py(e_idx,2)*t.^2 + py(e_idx,3)*t + py(e_idx,4);
        Pz = pz(e_idx,1)*t.^3 + pz(e_idx,2)*t.^2 + pz(e_idx,3)*t + pz(e_idx,4);

        status_1 = func_3D_SOS_Tube_spotless(Safe_1, t0, tf, Px, Py, Pz, es(:,e_idx), ec(1,e_idx,:));
        status_2 = func_3D_SOS_Tube_spotless(Safe_2, t0, tf, Px, Py, Pz, es(:,e_idx), ec(1,e_idx,:));

        statuses(e_idx) = (status_1 == 1 && status_2 == 1);
    end
    status = all(statuses);
    disp(['risk verification time: ',num2str(toc)]);
    end
end



function status=func_3D_SOS_Tube_spotless(Safe,Px,Py,Pz,t0,tf,R,d, lc)
    % SOS condition: Safe(x(t)+xt,y(t)+yt,z(t)+zt)>=0 for all t0 =<t=< tf and (xt,yt) in {R^2-xt^2-yt^2-zt^2} 
    t = msspoly('t',1);
    xt = msspoly('xt',1);
    yt = msspoly('yt',1);
    zt = msspoly('zt',1);
    
    prog = spotsosprog;
    prog = prog.withIndeterminate( [t,xt,yt,zt] );
    [prog,gamma] = prog.newFree(1);
    
    bases = monomials( [t,xt,yt,zt], 0:d );
    
    [ prog, s1 ] = prog.newFreePoly( bases ); 
    prog = prog.withSOS(s1);
    
    [ prog, s2 ] = prog.newFreePoly( bases ); 
    prog = prog.withSOS(s2);
    prog = prog.withSOS( Safe(Px+xt,Py+yt,Pz+zt,t)-gamma -s1*(t-t0)*(tf-t)-s2*(R^2-(xt-lc(1))^2-(yt-lc(2))^2-(zt-lc(3))^2)) ); 
    
    spot_options = spot_sdp_default_options();
    spot_options.verbose = 0; % printing information
    sol = prog.minimize(-gamma, @spot_mosek,spot_options);
    
    
    if double(sol.eval(gamma)) >=0 
        status = 1
    else
        status = 0
        display('Trajectory is NOT safe.') 
    end
end


