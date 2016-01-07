clear;
% test mex
rng(0823);
XPts = 100;
XGrid = linspace(0,1,XPts);
dim = 4;
Y = rand(dim,XPts);

pp = struct('form','MKLpp','breaks',{XGrid},...
    'Values',reshape(Y,[],XPts),'coefs',[],'order',[2],...
    'Method',[],'ExtrapolationOrder',[],'thread',1,...
    'orient','curvefit');
pp = myppual(pp);
ppCoefs = pp.coefs;
ppOrder = double(pp.order);

SitePts = 10000;
XSite = rand(1,SitePts);
Idx = ceil(dim*rand(1,SitePts));
YSite = zeros(1,SitePts);

YSitePp = myppual(pp,XSite,[],Idx);

interpMex;

Err = max(abs(YSite-YSitePp))