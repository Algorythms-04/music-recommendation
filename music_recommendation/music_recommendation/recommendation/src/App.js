import React, { useState, useEffect, useRef } from 'react';

// Song database for offline demo
const songDatabase = {
  "bohemian rhapsody": [
    { title: "Stairway to Heaven", artist: "Led Zeppelin", genre: "Rock" },
    { title: "November Rain", artist: "Guns N' Roses", genre: "Rock" },
    { title: "Hotel California", artist: "Eagles", genre: "Classic Rock" },
    { title: "Comfortably Numb", artist: "Pink Floyd", genre: "Progressive Rock" }
  ],
  "billie jean": [
    { title: "Smooth Criminal", artist: "Michael Jackson", genre: "Pop" },
    { title: "Beat It", artist: "Michael Jackson", genre: "Pop" },
    { title: "When Doves Cry", artist: "Prince", genre: "Pop" },
    { title: "Stayin' Alive", artist: "Bee Gees", genre: "Disco" }
  ],
  "shape of you": [
    { title: "Señorita", artist: "Shawn Mendes & Camila Cabello", genre: "Pop" },
    { title: "Blinding Lights", artist: "The Weeknd", genre: "Synth-Pop" },
    { title: "Dance Monkey", artist: "Tones and I", genre: "Pop" },
    { title: "Watermelon Sugar", artist: "Harry Styles", genre: "Pop Rock" }
  ]
};

// Default recommendations
const defaultRecommendations = [
  { title: "Bad Guy", artist: "Billie Eilish", genre: "Alt Pop" },
  { title: "Blinding Lights", artist: "The Weeknd", genre: "Synth-Pop" },
  { title: "Levitating", artist: "Dua Lipa", genre: "Dance Pop" },
  { title: "Dynamite", artist: "BTS", genre: "K-Pop" }
];

// Custom SVG icons as components
const MusicIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 18V5l12-2v13"></path>
    <circle cx="6" cy="18" r="3"></circle>
    <circle cx="18" cy="16" r="3"></circle>
  </svg>
);

const SearchIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
);

const DiscIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <circle cx="12" cy="12" r="3"></circle>
  </svg>
);

const PlayIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="5 3 19 12 5 21 5 3"></polygon>
  </svg>
);

// Add script loader utilities
const loadScript = (src) => {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

const SoundWave = () => {
  // State variables
  const [songInput, setSongInput] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [animateWave, setAnimateWave] = useState(false);
  const [listeningNowText, setListeningNowText] = useState('Discover your next favorite song');
  
  // Ref for the vanta background
  const vantaRef = useRef(null);
  const vantaEffect = useRef(null);
  const appRef = useRef(null);

  // Initialize Vanta.js effect
  useEffect(() => {
    const loadVantaEffect = async () => {
      try {
        // Load Three.js first
        await loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js');
        // Then load Vanta
        await loadScript('https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.halo.min.js');
        
        // Initialize Vanta effect if the libraries are loaded and the element exists
        if (window.VANTA && vantaRef.current && !vantaEffect.current) {
          vantaEffect.current = window.VANTA.HALO({
            el: vantaRef.current,
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            amplitudeFactor: 3.00,
            size: 3.00
          });
        }
      } catch (error) {
        console.error("Failed to load Vanta.js:", error);
      }
    };

    loadVantaEffect();

    // Cleanup
    return () => {
      if (vantaEffect.current) {
        vantaEffect.current.destroy();
      }
    };
  }, []);

  // Generate gradient colors for card backgrounds
  const getGradient = () => {
    const colors = [
      '#ff6b6b', '#ff8e53', '#ffb347', '#48dbfb', '#0abde3', 
      '#55efc4', '#00b894', '#ffeaa7', '#fab1a0', '#fd79a8',
      '#a29bfe', '#6c5ce7', '#74b9ff', '#0984e3', '#badc58'
    ];
    
    const color1 = colors[Math.floor(Math.random() * colors.length)];
    const color2 = colors[Math.floor(Math.random() * colors.length)];
    
    return `linear-gradient(45deg, ${color1}, ${color2})`;
  };

  // Search songs function (tries API first, falls back to local DB)
  const searchSongs = async () => {
    const songName = songInput.trim().toLowerCase();
    
    if (!songName) {
      setError('Please enter a song name');
      setRecommendations([]);
      return;
    }
    
    setError('');
    setIsLoading(true);
    setAnimateWave(true);
    
    try {
      // Try to fetch from API first
      const response = await fetch(`http://localhost:8000/recommend?track_name=${encodeURIComponent(songName)}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        // Add a timeout to fail faster if API is not available
        signal: AbortSignal.timeout(3000)
      });
      
      if (!response.ok) {
        throw new Error('Server error');
      }
      
      const data = await response.json();
      
      if (data && data.recommendations) {
        const formattedRecommendations = data.recommendations.map(song => ({
          title: song.track_name || 'Unknown Song',
          artist: song.artists || 'Unknown Artist',
          genre: song.genre || 'Unknown Genre'
        }));
        
        setRecommendations(formattedRecommendations);
        
        const capitalizedSongName = data.track_name || songName;
        setListeningNowText(`Based on "${capitalizedSongName}", we think you'll love these:`);
      } else {
        throw new Error('Invalid API response');
      }
    } catch (error) {
      console.log('Falling back to local database');
      
      // If API fails, use local database
      setTimeout(() => {
        // Check if song exists in our local database
        if (songDatabase[songName]) {
          setRecommendations(songDatabase[songName]);
          setListeningNowText(`Based on "${songName}", we think you'll love these:`);
        } else {
          setRecommendations(defaultRecommendations);
          setListeningNowText("Try another song for better recommendations!");
        }
      }, 1000);
    } finally {
      setTimeout(() => {
        setIsLoading(false);
      }, 1500);
    }
  };

  // Handle song card click
  const handleSongCardClick = (song) => {
    setSongInput(song.title);
    // Use setTimeout to ensure the input is updated before searching
    setTimeout(() => {
      searchSongs();
    }, 100);
  };

  // Handle key press for searching
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      searchSongs();
    }
  };

  return (
    <div 
      ref={vantaRef}
      style={{
        minHeight: "100vh",
        height: "100%",
        width: "100%",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "20px",
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
      }}
    >
      <div 
        ref={appRef}
        style={{
          width: "100%",
          maxWidth: "600px",
          backgroundColor: "rgba(0, 0, 0, 0.3)",
          backdropFilter: "blur(10px)",
          borderRadius: "24px",
          boxShadow: "0 10px 30px rgba(0, 0, 0, 0.2)",
          padding: "32px",
          transform: "scale(1)",
          transition: "all 0.5s",
          marginTop: "20px",
          marginBottom: "20px",
        }}
      >
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "24px",
        }}>
          <div style={{ position: "relative", marginRight: "16px" }}>
            <div style={{
              width: "48px",
              height: "48px",
              color: "white",
              animation: "pulse 2s infinite",
            }}>
              <MusicIcon />
            </div>
            <div style={{
              position: "absolute",
              top: "-4px",
              right: "-4px",
              width: "16px",
              height: "16px",
              backgroundColor: "#f06",
              borderRadius: "50%",
              animation: "ping 1s infinite",
            }} />
          </div>
          <h1 style={{
            fontSize: "3rem",
            fontWeight: "bold",
            background: "linear-gradient(to right, #f06, #9f7aea)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>SoundWave</h1>
        </div>

        <div style={{
          display: "flex",
          marginBottom: "24px",
          position: "relative",
        }}>
          <input 
            type="text" 
            value={songInput}
            onChange={(e) => setSongInput(e.target.value)}
            onKeyUp={handleKeyPress}
            placeholder="Enter a song you like..." 
            style={{
              flexGrow: "1",
              padding: "16px 24px",
              borderRadius: "999px",
              backgroundColor: "rgba(255, 255, 255, 0.1)",
              color: "white",
              fontSize: "1.1rem",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              outline: "none",
            }}
          />
          <button 
            onClick={searchSongs} 
            style={{
              position: "absolute",
              right: "4px",
              top: "4px",
              padding: "12px 24px",
              background: "linear-gradient(to right, #9f7aea, #f06)",
              color: "white",
              borderRadius: "999px",
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s",
            }}
          >
            <SearchIcon />
          </button>
        </div>

        {error && (
          <div style={{
            backgroundColor: "rgba(255, 0, 0, 0.2)",
            color: "white",
            padding: "16px",
            borderRadius: "12px",
            textAlign: "center",
            marginBottom: "16px",
            border: "1px solid rgba(255, 0, 0, 0.3)",
            animation: "pulse 2s infinite",
          }}>
            {error}
          </div>
        )}

        {isLoading && (
          <div style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: "12px",
            padding: "40px 0",
          }}>
            {[...Array(7)].map((_, index) => (
              <div 
                key={index} 
                style={{ 
                  width: "8px",
                  height: "64px",
                  background: "linear-gradient(to top, #9f7aea, #f06)",
                  borderRadius: "999px",
                  animation: "sound-wave 1s infinite",
                  animationDelay: `${index * 0.1}s`,
                  animationDuration: `${0.8 + Math.random() * 0.4}s`,
                }}
              />
            ))}
          </div>
        )}

        {recommendations.length > 0 && !isLoading && (
          <div style={{ marginTop: "16px" }}>
            <h2 style={{
              fontSize: "1.5rem",
              fontWeight: "600",
              background: "linear-gradient(to right, #f06, #60a5fa)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              textAlign: "center",
              marginBottom: "24px",
            }}>
              {listeningNowText}
            </h2>
            <div style={{
              display: "grid",
              gap: "16px",
            }}>
              {recommendations.map((song, index) => (
                <div 
                  key={index} 
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.1)",
                    borderRadius: "16px",
                    padding: "16px",
                    display: "flex",
                    alignItems: "center",
                    gap: "16px",
                    cursor: "pointer",
                    transition: "all 0.3s",
                    border: "1px solid rgba(255, 255, 255, 0.05)",
                  }}
                  onClick={() => handleSongCardClick(song)}
                  onMouseOver={(e) => e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.15)"}
                  onMouseOut={(e) => e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.1)"}
                >
                  <div style={{
                    width: "64px",
                    height: "64px",
                    borderRadius: "12px",
                    background: getGradient(),
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    position: "relative",
                  }}>
                    <div className="disc-icon" style={{ color: "rgba(255, 255, 255, 0.8)" }}>
                      <DiscIcon />
                    </div>
                    <div className="play-icon" style={{ 
                      color: "white", 
                      display: "none",
                      position: "absolute"
                    }}>
                      <PlayIcon />
                    </div>
                  </div>
                  <div style={{ flexGrow: "1" }}>
                    <h3 style={{
                      fontSize: "1.25rem",
                      fontWeight: "bold",
                      color: "white",
                    }}>{song.title}</h3>
                    <p style={{
                      color: "rgba(255, 255, 255, 0.7)",
                      fontSize: "0.9rem",
                    }}>{song.artist}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!isLoading && recommendations.length === 0 && !error && (
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "32px 0",
            color: "rgba(255, 255, 255, 0.7)",
          }}>
            <div style={{
              width: "64px",
              height: "64px",
              marginBottom: "16px",
              animation: "pulse 2s infinite",
            }}>
              <MusicIcon />
            </div>
            <p style={{ textAlign: "center" }}>Discover new music by entering a song you love</p>
          </div>
        )}

        <style>
          {`
            @keyframes sound-wave {
              0%, 100% { height: 10px; }
              50% { height: 64px; }
            }
            
            @keyframes pulse {
              0%, 100% { transform: scale(1); }
              50% { transform: scale(1.1); }
            }
            
            @keyframes ping {
              0% { transform: scale(1); opacity: 1; }
              75%, 100% { transform: scale(2); opacity: 0; }
            }
            
            div:hover .disc-icon {
              display: none;
            }
            
            div:hover .play-icon {
              display: block;
            }
            
            /* Body and html styles to ensure background extends */
            html, body {
              margin: 0;
              padding: 0;
              height: 100%;
              width: 100%;
              overflow-x: hidden;
              background: #000;
              position: relative;
            }
            
            #root {
              min-height: 100vh;
              position: relative;
            }
          `}
        </style>
      </div>
    </div>
  );
};

export default SoundWave;