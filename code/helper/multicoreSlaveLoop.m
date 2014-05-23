function  multicoreSlaveLoop( )
%MULTICORESLAVELOOP Start multicore slave with try catch block to resume
%being slave

multicoreDir='/nfs/nhome/live/wittawat/SHARE/gatsby/research/code/tmp';
for i=1:5
    try
        startmulticoreslave(multicoreDir);
    catch errors
        pause(5);
    end
    
end

end


