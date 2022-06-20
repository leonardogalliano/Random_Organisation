using LinearAlgebra, Distributions, FLoops

mutable struct Parameters{T <: AbstractFloat}
    N::Int      #number of particles
    M::Int      #number of measures
    L::T        #length of the box
    σ::T        #particle diameter
    ϵ::T        #maximum size of the kick
    κ::T        #blending between BRO and RO
    η::T        #polydispersity
    Nb::Int     #number of cells per side linkedlist
end

function Parameters(N::Int, M::Int, L::T, ϵ::T; σ::T = one(T), κ::T = one(T),
        η::T = zero(T)) where T <: AbstractFloat
    return Parameters(N, M, L, σ, ϵ, κ, η, Int(fld(L, σ)))
end

mutable struct Config{T <: AbstractFloat}
    pos::Vector{Tuple{T, T}}    #position of all particles
    active::Vector{Bool}        #list of active particles
    sizes::Vector{T}            #sizes of the particle
end

mutable struct Measures{T <: AbstractFloat}
    pos::Vector{Vector{Tuple{T, T}}}    #stored positions
    active::Vector{T}                   #stored order parameter
    sampletimes::Vector{Int}            #time instant of each measurement
end

#Method for dynamics
abstract type Model end
struct RO <: Model end
struct BRO <: Model end

confacfrac(conf::Config) = sum(conf.active) / length(conf.pos)

function pbcdist(x₁::T, x₂::T, L::T) where T <: AbstractFloat
    if x₁ - x₂ > L / 2.0
        return x₁ - x₂ - L
    elseif x₁ - x₂ < - L / 2.0
        return x₁ - x₂ + L
    else
        return x₁ - x₂
    end
end

function dist(p₁::Tuple{T, T}, p₂::Tuple{T, T}, L::T) where T <: AbstractFloat
    return (pbcdist(p₁[1], p₂[1], L), pbcdist(p₁[2], p₂[2], L))
end

function deltafill!(𝜹::Vector{Tuple{T, T}}, conf::Config, L::T, i::Int, j::Int,
    ::Type{RO}) where T <: AbstractFloat
    #Random kick
    𝜹[i], 𝜹[j] = map(randn, (T, T)), map(randn, (T, T))
    nothing
end

function deltafill!(𝜹::Vector{Tuple{T, T}}, conf::Config, L::T, i::Int, j::Int,
    ::Type{BRO}) where T <: AbstractFloat
    #Evaluate distance (with PBC)
    dij = dist(conf.pos[i], conf.pos[j], L)
    #Update 𝜹
    𝜹[i] = 𝜹[i] .+ dij
    𝜹[j] = 𝜹[j] .- dij
    nothing
end

function linkedlist(𝜹::Vector{Tuple{T, T}}, conf::Config, par::Parameters,
        head::Vector{Int}, lscl::Vector{Int},
        model::Type{W} = RO) where T <: AbstractFloat where W <: Model
    #Define parameters
    N = par.N
    Nb = par.Nb
    L = par.L
    mc, mc1 = [1, 1], [1, 1]
    rc = L / Nb
    rshift = zeros(T, 2)

    #LIST CONSTRUCTOR
    #Reset the headers
    @inbounds for c ∈ 1 : Nb^2
        head[c] = -1
    end
    #Scan atoms to construct headers and linked lists
    @inbounds for i ∈ 1 : N
        #Vector cell index to which this atom belongs
        mc[1], mc[2] = Int(fld(conf.pos[i][1], rc)), Int(fld(conf.pos[i][2], rc))
        #Translate the vector cell index, mc, to a scalar cell index
        c = Nb * mc[1] + mc[2] + 1
        #Link to the previous occupant (or EMPTY (-1) if you're the 1st)
        lscl[i] = head[c]
        #The last one goes to the header
        head[c] = i
    end

    #INTERACTION COMPUTATION
    #Scan inner cells
    @inbounds for mc[1] ∈ 0 : Nb - 1, mc[2] ∈ 0 : Nb - 1
        #Calculate a scalar index
        c = Nb * mc[1] + mc[2] + 1
        #Scan the neighbor cells (including itself) of cell c
        @inbounds for mc1[1] ∈ mc[1] - 1 : mc[1] + 1, mc1[2] ∈ mc[2] - 1 : mc[2] + 1
            #Periodic boundary condition by shifting coordinates
            @inbounds for a ∈ 1 : 2
                if mc1[a] < 0
                    rshift[a] = -L
                elseif mc1[a] ≥ Nb
                    rshift[a] = L
                else
                    rshift[a] = zero(T)
                end
            end
            #Calculate the scalar cell index of the neighbor cell (with PBC)
            c1 = ((mc1[1] + Nb) % Nb) * Nb + ((mc1[2] + Nb) % Nb) + 1
            #Scan atom i in cell c
            i = head[c]
            while (i != -1)
                #Scan atom j in cell c1
                j = head[c1]
                while (j != -1)
                    #Avoid double counting
                    if i < j
                        #Define distace between particles
                        Δr = conf.pos[i] .- (conf.pos[j] .+ (rshift[1], rshift[2]))
                        #Check for collisions
                        if norm(Δr) ≤ (conf.sizes[i] + conf.sizes[j]) / 2.0
                            #Update 𝜹 for active particles
                            deltafill!(𝜹, conf, par.L, i, j, model)
                        end
                    end
                    j = lscl[j]
                end
                i = lscl[i]
            end
        end
    end

    #REGULARISATION
    #Check for active particles
    active = findall(a -> !iszero(a), sum.(𝜹))
    #Random blend option
    if !isone(par.κ) && model == BRO
        @inbounds for i ∈ active
            𝜹[i] = par.κ .* 𝜹[i] .+ (1 - par.κ) .* map(randn, (T, T))
        end
    end
    #Normalisation
    @inbounds for i ∈ active
        𝜹[i] = rand(Uniform(0.0, par.ϵ)) .* 𝜹[i] ./ norm(𝜹[i])
    end

    #Return list of actives
    return active
end

function initialconf(par::Parameters, 𝐗₀::Vector{Tuple{T, T}},
        𝜹::Vector{Tuple{T, T}}, head::Vector{Int}, lscl::Vector{Int},
        model::Type{W} = RO) where T <: AbstractFloat where W <: Model
    #Generate sizes
    N = par.N
    sizes = rand(Normal(par.σ, sqrt(par.η * par.σ)), N)
    #Initialisation
    conf = Config(fill((zero(T), zero(T)), N), zeros(Bool, N), sizes)
    conf.pos = 𝐗₀
    #Check for collisions
    active = linkedlist(𝜹, conf, par, head, lscl, model)
    conf.active[active] .= true
    return conf
end

function initialmeas(par::Parameters, conf::Config, sampletimes::Vector{Int},
    Mp::Int, T::DataType)
    #Initialisation
    N, M = par.N, par.M
    meas = Measures(fill(fill((zero(T), zero(T)), N), Mp), zeros(T, M), sampletimes)
    #First measure
    meas.active[1] = confacfrac(conf)
    meas.pos[1] = copy(conf.pos)
    return meas
end

function update!(conf::Config, par::Parameters, 𝜹::Vector{Tuple{T, T}},
        head::Vector{Int}, lscl::Vector{Int},
        model::Type{W} = RO) where T <: AbstractFloat where W <: Model
    #Reset 𝜹
    N = par.N
    @inbounds for i ∈ 1 : N
        𝜹[i] = (zero(T), zero(T))
    end
    #Check for collisions
    active = linkedlist(𝜹, conf, par, head, lscl, model)
    #Update all positions (with PBC)
    X₂ = [conf.pos[i] .+ 𝜹[i] for i ∈ 1 : N]
    conf.pos = [X₂[i] .- fld.(X₂[i], par.L) .* par.L for i ∈ 1 : N]
    #Update actives
    conf.active .= false
    conf.active[active] .= true
    nothing
end

function whentomeasure(t₀::Int, tmax::Int, nmeas::Int;
        slope::T = 1.0) where T <: AbstractFloat
    #Create sampletimes and fix extrema
    sampletimes = zeros(Int, nmeas)
    sampletimes[end], sampletimes[1] = t₀, tmax
    #Power law sampling
    rng = LinRange(0, 1, nmeas)
    for m ∈ 2 : nmeas - 1
        sampletimes[m] =  Int(floor(-rng[m]^slope * (tmax - t₀) + tmax))
    end
    #Finishing
    return unique!(reverse!(sampletimes))
end

function simulation(N::Int,     #number of particles
        tmax::Int,              #number of time steps
        L::T;                   #length of the box
        t₀::Int = 0,            #termalisation time
        nmeas::Int = Int(floor(log(2, (tmax - t₀)))), #number of measurements (approximate)
        slope::T = 1.0,         #slope of the sampling
        Δm::Int = 1,            #intervals to store full configurations
        sampletimes::Vector{Int} = whentomeasure(t₀, tmax, nmeas, slope = slope), #optional custom sampletimes
        σ::T = 1.0,             #particle diameter
        ϵ::T = 0.5,             #maximum size of the kick
        κ::T = 1.0,             #blending between BRO and RO
        η::T = 0.0,             #polydispersity
        𝐗₀::Vector{Tuple{T, T}} = [L .* map(rand, (T, T)) for i ∈ 1 : N], #initial configuration
        model::Type{W} = RO) where T <: AbstractFloat where W <: Model

    #INISIALISATION
    #Define parameters
    M = length(sampletimes)
    Mp = length(1 : Δm : M)
    par = Parameters(N, M, L, ϵ, σ = σ, κ = κ, η = η)
    Δt = [sampletimes[m] - sampletimes[m - 1] for m ∈ 2 : M]
    Δm = M ÷ Mp
    #Initialse arrays
    head = - ones(Int, par.Nb^2)
    lscl = - ones(Int, N)
    𝜹 = [(zero(T), zero(T)) for i ∈ 1 : N]
    conf = initialconf(par, 𝐗₀, 𝜹, head, lscl, model)
    #Update until termalisation
    @inbounds for t ∈ 1 : t₀
        update!(conf, par, 𝜹, head, lscl, model)
    end
    #First measurement
    meas = initialmeas(par, conf, sampletimes, Mp, T)

    #MAIN LOOP
    mp = 2
    @inbounds for m ∈ 2 : M
        #Update between two measurements
        @inbounds for t ∈ 1 : Δt[m - 1]
            update!(conf, par, 𝜹, head, lscl, model)
            #Break in case of absorbing state
            iszero(confacfrac(conf)) && break
        end
        #Measure active particles
        meas.active[m] = confacfrac(conf)
        #Store complete configuration once every Δm
        if iszero((m - 1) % Δm)
            meas.pos[mp] = copy(conf.pos)
            mp += 1
        end
        #Fill in case of absorbing state
        if iszero(confacfrac(conf))
            meas.pos[mp : Mp] .= [meas.pos[mp - 1]]
            break
        end
    end
    #Store the last configuration
    meas.pos[Mp] = copy(conf.pos)

    return meas

end

function runs(n::Int,   #number of simulations
        N::Int,         #number of particles
        tmax::Int,      #number of time steps
        L::T;           #length of the box
        t₀::Int = 0,    #termalisation time
        nmeas::Int = Int(floor(log(2, (tmax - t₀)))), #number of measurements (approximate)
        Δm::Int = 10,   #intervals to store full configurations
        sampletimes::Vector{Int} = whentomeasure(t₀, tmax, nmeas), #optional custom sampletimes
        σ::T = 1.0,     #particle diameter
        ϵ::T = 0.5,     #maximum size of the kick
        κ::T = 1.0,     #blending between BRO and RO
        η::T = 0.0,     #polydispersity
        𝐗₀::Vector{Tuple{T, T}} = [L .* map(rand, (T, T)) for i ∈ 1 : N], #initial configuration
        model::Type{W} = RO,    #dynamical rule
        verbose::Bool = false,  #debug
        ncores::Int = Threads.nthreads()) where T <: AbstractFloat where W <: Model

    res = Vector{Measures}(undef, n)
    @floop ThreadedEx(basesize = n ÷ ncores) for k ∈ 1 : n
        res[k] = simulation(N, tmax, L, t₀ = t₀, nmeas = nmeas, Δm = Δm, sampletimes = sampletimes, σ = σ, ϵ = ϵ, κ = κ, η = η, 𝐗₀ = 𝐗₀, model = model)
        verbose && println("sim = $k on thread $(Threads.threadid())")
    end
    return res

end
nothing
