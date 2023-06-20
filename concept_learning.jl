using Distributed
using HypothesisTests
using GLM
using EffectSizes

addprocs(...) # set number of cores for parallel processing

@everywhere begin
    using CSV
    using Colors
    using DataFrames
    using Distributions
    using LinearAlgebra
    using Distances
    using Clustering
    using ClusterAnalysis
    using StatsBase
    using MLJBase
    using NearestNeighborModels
end

@everywhere function luv_convert(crds, i)
	c = convert(Luv, RGB(crds[i, :]...))
	return c.l, c.u, c.v
end

#= to run the study in CIELAB, everywhere use this function instead of the one above:
@everywhere function lab_convert(crds, i)
	c = convert(Lab, RGB(crds[i, :]...))
	return c.l, c.a, c.b
end
=#

@everywhere coords_full = CSV.read("/home/igor/Projects/Carnap_Analogy/munsell_rgb.csv", DataFrame) |> Matrix
@everywhere luv_coords_full = [ luv_convert(coords_full, i) for i in 1:size(coords_full, 1) ]

@everywhere coords = CSV.read(".../rgb320.csv", DataFrame; header=false)./255 |> Matrix
@everywhere luv_coords = [ luv_convert(coords, i) for i in 1:size(coords, 1) ]

@everywhere luv_df = DataFrame(luv_coords)

# run the experiments both for 10 and 11 clusters, given that we are leaving out the achromatic chips (and gray is sometimes not represented in the naming data for the 320 chromatic WCS chips)
@everywhere label10(v::Vector{Vector{Float64}}) = [ findmin([ Distances.evaluate(Euclidean(), v[i], luv_df[j, :]) for i in 1:10 ])[2] for j in 1:size(luv_df, 1) ]
@everywhere label11(v::Vector{Vector{Float64}}) = [ findmin([ Distances.evaluate(Euclidean(), v[i], luv_df[j, :]) for i in 1:11 ])[2] for j in 1:size(luv_df, 1) ]

@everywhere ca10 = ClusterAnalysis.kmeans(luv_df, 10; nstart=20, maxiter=25)
@everywhere ca11 = ClusterAnalysis.kmeans(luv_df, 11; nstart=20, maxiter=25)

@everywhere cl10 = [ vcat(collect.([luv_coords_full[sample(1:1625, 10; replace=false)]...])) for _ in 1:20_000 ]
@everywhere rand_clust10 = label10.(cl10)
# sometimes the above procedure will not result in 10 categories, so we create twice as many random clusterings as needed and select 1000 with the required number of clusters
@everywhere random_clustering10 = rand_clust10[length.(unique.(rand_clust10)) .== 10][1:10_000]
for_calc10 = cl10[length.(unique.(rand_clust10)) .== 10][1:10_000]

@everywhere cl11 = [ vcat(collect.([luv_coords_full[sample(1:1625, 11; replace=false)]...])) for _ in 1:20_000 ]
@everywhere rand_clust11 = label11.(cl11)
@everywhere random_clustering11 = rand_clust11[length.(unique.(rand_clust11)) .== 11][1:10_000]
for_calc11 = cl11[length.(unique.(rand_clust11)) .== 11][1:10_000]

#######################################
## Approximate Prototype Model ##
#######################################

# we give a clustering with 10/11 clusters as input, then sample from each cell a random number of items, where the
# number can be different for each cell; the means of the sample for the cells then serve as best guesses of focal colors,
# which are used to determine a clustering (assign chips to closest best guess); finally, we compare the similarity of 
# the resulting 'guessed' clustering with the input clustering 
@everywhere function compare_clusterings(labs::Vector{Int64}; numb_clust::Int=11)
    df = copy(luv_df)
    df.lbs = labs
    df_gb = groupby(df, :lbs)
    df_samp = reduce(vcat, [ df_gb[i][sample(axes(df_gb[i], 1), rand(axes(df_gb[i], 1)); replace=false, ordered=true), :] for i in 1:numb_clust ])
    df_samp_gb = groupby(df_samp, :lbs)
    cb = combine(df_samp_gb, valuecols(df_samp_gb) .=> mean)
    labs_estimate = numb_clust == 11 ? label11(vcat([ Matrix(cb)[i, 2:4] for i in 1:numb_clust ])) : label10(vcat([ Matrix(cb)[i, 2:4] for i in 1:numb_clust ]))
    return mutualinfo(labs, labs_estimate)
end

berlin_kay10 = [ compare_clusterings(ca10.cluster; numb_clust=10) for _ in 1:50 ]
berlin_kay11 = [ compare_clusterings(ca11.cluster) for _ in 1:50 ]

mean_and_std(berlin_kay10)
extrema(berlin_kay10)
mean_and_std(berlin_kay11)
extrema(berlin_kay11)

res10 = pmap(i->mean_and_std([ compare_clusterings(random_clustering10[i]; numb_clust=10) for _ in 1:50 ]), 1:length(random_clustering10))
res11 = pmap(i->mean_and_std([ compare_clusterings(random_clustering11[i]) for _ in 1:50 ]), 1:length(random_clustering11))

first.(res10) |> mean_and_std
first.(res11) |> mean_and_std
extrema(first.(res10))
extrema(first.(res11))

# regress accuracy on contrast and representation

function calc(ar::Vector{Vector{Float64}}) # calculates contrast and representation for a constellation
    c = sum(LowerTriangular(pairwise(Euclidean(), reduce(hcat, ar))))
    find_color(x) = [ findmin([ Distances.evaluate(Euclidean(), x[i], [luv_coords[j]...]) for i in 1:length(ar) ])[2] for j in 1:size(luv_coords, 1) ]
    fc = find_color(ar)
    dfg = DataFrame(first=first.(luv_coords), second=getindex.(luv_coords, 2), third=last.(luv_coords), group=fc)
    gd = groupby(dfg, :group)
    cd = combine(gd, valuecols(gd) .=> mean)
    centers = Matrix(cd[:, 2:4]) #convert(Matrix{Float32}, Matrix(cd[:, 2:4]))
    r = 0.0
    for i in 1:length(ar)
        r += euclidean(centers[i, :], reduce(hcat, ar)'[i, :])
    end
    return [c, r]
end

cr10 = calc.(for_calc10)
cr11 = calc.(for_calc11)

function reg_mod(v::Vector{Tuple{Float64, Float64}}, c::Vector{Vector{Float64}})
    dv=StatsBase.fit(ZScoreTransform, first.(v)) # we're obtaining β coefficients
    iv1=StatsBase.fit(ZScoreTransform, first.(c))
    iv2=StatsBase.fit(ZScoreTransform, last.(c))
    df = DataFrame(DV=StatsBase.transform(dv, first.(v)), IV1=StatsBase.transform(iv1, first.(c)), IV2=StatsBase.transform(iv2, last.(c)))
    mod = lm(@formula(DV ~ IV1 + IV2), df) # lm(@formula(DV ~ IV1 * IV2), df) no significant interaction
    return mod
end

reg_mod(res10, cr10)
reg_mod(res11, cr11)

#########
## KNN ##
#########

@everywhere function knn_compare(labs::Vector{Int64}; numb_clust::Int=11)
    df = copy(luv_df)
    df.lbs = labs
    df_gb = groupby(df, :lbs)
    df_samp = reduce(vcat, [ df_gb[i][sample(axes(df_gb[i], 1), rand(axes(df_gb[i], 1)); replace=false, ordered=true), :] for i in 1:numb_clust ])
    X = (L = df_samp[:, 1], u = df_samp[:, 2], v = df_samp[:, 3])
    Y = categorical(df_samp.lbs)
    knn = KNNClassifier(K=round(Int, sqrt(nrow(df_samp))), weights=ISquared())
    knn_mach = machine(knn, X, Y)
    fit!(knn_mach, verbosity=0)
    full_crd = (L = luv_df[:, 1], u = luv_df[:, 2], v = luv_df[:, 3])
    labs_estimate = predict_mode(knn_mach, full_crd)
    return numb_clust==11 ? mutualinfo(ca11.cluster, convert.(Int, labs_estimate)) : mutualinfo(ca10.cluster, convert.(Int, labs_estimate))
end

berlin_kay_knn10 = [ knn_compare(ca10.cluster; numb_clust=10) for _ in 1:50 ]
berlin_kay_knn11 = [ knn_compare(ca11.cluster) for _ in 1:50 ]

mean_and_std(berlin_kay_knn10)
extrema(berlin_kay_knn10)
mean_and_std(berlin_kay_knn11)
extrema(berlin_kay_knn11)

res_knn10 = pmap(i->mean_and_std([ knn_compare(random_clustering10[i]; numb_clust=10) for _ in 1:50 ]), 1:length(random_clustering10))
res_knn11 = pmap(i->mean_and_std([ knn_compare(random_clustering11[i]) for _ in 1:50 ]), 1:length(random_clustering11))

first.(res_knn10) |> mean_and_std
first.(res_knn11) |> mean_and_std

extrema(first.(res_knn10))
extrema(first.(res_knn11))

# same regression analysis as above
reg_mod(res_knn10, cr10)
reg_mod(res_knn11, cr11)

# compare APM and KNN
EqualVarianceTTest(berlin_kay10, berlin_kay_knn10)
EqualVarianceTTest(berlin_kay11, berlin_kay_knn11)
EqualVarianceTTest(first.(res10), first.(res_knn10))
EqualVarianceTTest(first.(res11), first.(res_knn11))

CohenD(berlin_kay10, berlin_kay_knn10)
CohenD(berlin_kay11, berlin_kay_knn11)
CohenD(first.(res10), first.(res_knn10))
CohenD(first.(res11), first.(res_knn11))

