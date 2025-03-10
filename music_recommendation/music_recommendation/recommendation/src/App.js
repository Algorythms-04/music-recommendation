import React, { useState } from 'react';

const SoundWave = () => {
  // Default recommendations
  const defaultRecommendations = [
    { title: "Bad Guy", artist: "Billie Eilish", image: "default1" },
    { title: "Blinding Lights", artist: "The Weeknd", image: "default2" },
    { title: "Levitating", artist: "Dua Lipa", image: "default3" },
    { title: "Dynamite", artist: "BTS", image: "default4" }
  ];

  // State variables
  const [songInput, setSongInput] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showError, setShowError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [listeningNowText, setListeningNowText] = useState('Discover your next favorite song');

  // Generate gradient colors
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

  // Search songs via API
  const searchSongs = async () => {
    const songName = songInput.trim();
    
    // Show error if input is empty
    if (!songName) {
      setShowError(true);
      setErrorMessage('Please enter a song name to get recommendations.');
      setRecommendations([]);
      return;
    }
    
    // Hide error message
    setShowError(false);
    
    // Show loading
    setIsLoading(true);
    
    try {
      // Send GET request to your server
      const response = await fetch(`http://localhost:8000/recommend?track_name=${encodeURIComponent(songName)}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Server error');
      }
      
      const data = await response.json();
      
      // Format the received data according to your API's response structure
      if (data && data.recommendations) {
        const formattedRecommendations = data.recommendations.map(song => ({
          title: song.track_name || 'Unknown Song',
          artist: song.artists || 'Unknown Artist',
          image: `song${Math.floor(Math.random() * 20) + 1}`
        }));
        
        setRecommendations(formattedRecommendations);
        
        const capitalizedSongName = data.track_name || songName;
        setListeningNowText(`Based on "${capitalizedSongName}", we think you'll love these:`);
      } else {
        // Fallback to default recommendations if API response format is unexpected
        setRecommendations(defaultRecommendations);
        setListeningNowText('Try another song for better recommendations!');
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setShowError(true);
      setErrorMessage(error.message === 'Failed to fetch' ? 
        'Cannot connect to the server. Is it running on localhost:8000?' : 
        error.message);
      setRecommendations(defaultRecommendations);
      setListeningNowText('Server issue. Try these popular tracks instead:');
    } finally {
      setIsLoading(false);
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

  // Custom card styles
  const cardStyle = {
    maxWidth: "600px",
    width: "100%",
    margin: "0 auto",
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    borderRadius: "12px",
    backdropFilter: "blur(10px)",
    boxShadow: "0 4px 30px rgba(0, 0, 0, 0.1)",
    padding: "20px",
  };

  const cardHeaderStyle = {
    marginBottom: "20px",
    textAlign: "center",
  };

  const cardTitleStyle = {
    fontSize: "2.5rem",
    fontWeight: "bold",
    color: "white",
    marginBottom: "10px",
  };

  return (
    <div style={{
      background: "linear-gradient(to right, #6a11cb, #2575fc)",
      minHeight: "100vh",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      padding: "20px",
    }}>
      <div style={cardStyle}>
        <div style={cardHeaderStyle}>
          <h1 style={cardTitleStyle}>SoundWave</h1>
        </div>
        <div>
          <div style={{
            display: "flex",
            marginBottom: "24px",
          }}>
            <input 
              type="text" 
              value={songInput}
              onChange={(e) => setSongInput(e.target.value)}
              onKeyUp={(e) => e.key === 'Enter' && searchSongs()}
              placeholder="Enter a song you like..." 
              style={{
                flex: 1,
                padding: "12px 16px",
                fontSize: "1.1rem",
                borderRadius: "999px 0 0 999px",
                border: "none",
                outline: "none",
              }}
            />
            <button 
              onClick={searchSongs}
              style={{
                background: "linear-gradient(to right, #f5576c, #f093fb)",
                color: "white",
                padding: "12px 24px",
                borderRadius: "0 999px 999px 0",
                fontSize: "1.1rem",
                fontWeight: "600",
                border: "none",
                cursor: "pointer",
              }}
            >
              Find Music
            </button>
          </div>
          
          {showError && (
            <div style={{
              color: "#ff6b6b",
              textAlign: "center",
              backgroundColor: "rgba(255, 0, 0, 0.1)",
              padding: "12px",
              borderRadius: "8px",
              marginBottom: "16px",
            }}>
              {errorMessage}
            </div>
          )}
          
          {isLoading && (
            <div style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: "8px",
              margin: "20px 0",
            }}>
              {[1, 2, 3, 4, 5].map((wave) => (
                <div 
                  key={wave} 
                  style={{
                    width: "4px",
                    height: "20px",
                    backgroundColor: "white",
                    borderRadius: "4px",
                    animation: "pulse 1s ease-in-out infinite",
                    animationDelay: `${wave * 0.1}s`,
                  }}
                />
              ))}
            </div>
          )}
          
          {recommendations.length > 0 && (
            <div>
              <h2 style={{
                fontSize: "1.5rem",
                color: "white",
                marginBottom: "16px",
              }}>
                {listeningNowText}
              </h2>
              
              <div style={{
                display: "flex",
                flexDirection: "column",
                gap: "16px",
              }}>
                {recommendations.map((song, index) => (
                  <div 
                    key={index} 
                    style={{
                      backgroundColor: "rgba(255, 255, 255, 0.15)",
                      borderRadius: "8px",
                      padding: "16px",
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                      cursor: "pointer",
                      transition: "background 0.3s ease",
                    }}
                    onClick={() => handleSongCardClick(song)}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.25)"}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.15)"}
                  >
                    <div 
                      style={{
                        width: "64px",
                        height: "64px",
                        borderRadius: "8px",
                        overflow: "hidden",
                        background: getGradient(),
                      }}
                    />
                    <div>
                      <div style={{
                        fontSize: "1.25rem",
                        fontWeight: "600",
                        color: "white",
                      }}>{song.title}</div>
                      <div style={{
                        color: "rgba(255, 255, 255, 0.8)",
                      }}>{song.artist}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SoundWave;