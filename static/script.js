// Recipe Card View - Course 3 Lesson 8 Enhancement

// Global variables for view state - make sure they're accessible to window
window.currentViewMode = 'text'; // or 'card'
window.parsedRecipes = []; // Store parsed recipe data

// Initialize view controls for toggle checkbox and minimize button after DOM loads
document.addEventListener('DOMContentLoaded', function () {
  console.log('DOM loaded, initializing view controls');

  // TODO: Initialize the toggle view switch

  // TODO: Initialize view toggle buttons in the index.html template

  // TODO: Set up window control buttons to minimize the chat window

  // TODO: Set up window control buttons to close the chat window

  // Make chat window draggable
  const chatHeader = document.querySelector('.chatbot-header');
  if (chatHeader && chatContainer) {
    let isDragging = false;
    let dragOffsetX, dragOffsetY;

    // Create a separate draggable area in the header to avoid conflicts with buttons
    const headerArea = document.createElement('div');
    headerArea.className = 'draggable-area';
    headerArea.style.position = 'absolute';
    headerArea.style.top = '0';
    headerArea.style.left = '0';
    headerArea.style.right = '0';
    headerArea.style.height = '50px'; // Limit height to header area only
    headerArea.style.zIndex = '50'; // Below buttons but above other elements
    headerArea.style.cursor = 'grab';
    headerArea.style.pointerEvents = 'auto';
    chatHeader.style.position = 'relative'; // Ensure the header has position for absolute child
    chatHeader.appendChild(headerArea);

    // Start drag only on the specific draggable area
    headerArea.addEventListener('mousedown', function (e) {
      // Make sure we're not clicking a button
      if (e.target === headerArea) {
        isDragging = true;

        // Calculate the offset of the mouse cursor from the top-left corner of the chat container
        const chatRect = chatContainer.getBoundingClientRect();
        dragOffsetX = e.clientX - chatRect.left;
        dragOffsetY = e.clientY - chatRect.top;

        // Add a dragging class to the container
        chatContainer.classList.add('dragging');
        e.preventDefault(); // Prevent text selection during drag
      }
    });

    // Track mouse movement
    document.addEventListener('mousemove', function (e) {
      if (!isDragging) return;

      // Calculate new position based on mouse coordinates and the initial offset
      const newLeft = e.clientX - dragOffsetX;
      const newTop = e.clientY - dragOffsetY;

      // Apply new position
      chatContainer.style.position = 'fixed';
      chatContainer.style.left = newLeft + 'px';
      chatContainer.style.top = newTop + 'px';
      chatContainer.style.margin = '0';
    });

    // End drag
    document.addEventListener('mouseup', function () {
      if (isDragging) {
        isDragging = false;
        chatContainer.classList.remove('dragging');
      }
    });
  }

  // TODO: Set up chat agent button in header. Listen for clicks to show chat window and hide button in header

  // Override the default sendQuery function with our enhanced version
  // Function also exists in the index.html template
  // Function handles streaming responses, spinner display, and recipe detection
  console.log('Overriding sendQuery function');
  const originalSendQuery = window.sendQuery;

  window.sendQuery = function () {
    const queryInput = document.getElementById("query");
    const query = queryInput.value.trim();
    if (!query) return;

    const chatWindow = document.getElementById("chat-window");

    // Append user's message
    chatWindow.innerHTML += `
        <div class="user-message">
          ${query}
        </div>
      `;
    queryInput.value = "";

    // Add spinner
    const spinnerId = "assistant-spinner-" + Date.now();
    chatWindow.innerHTML += `
        <div class="assistant-message" id="${spinnerId}">
          <div class="spinner"></div>
        </div>
      `;
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Prepare variables for streaming
    let partialResponse = "";          // Accumulate raw content without displaying
    let isFirstChunk = true;          // Track if first chunk
    let assistantResponseEl = null;   // Will be created but kept hidden until fully ready
    let pendingFormatting = true;     // Flag to prevent premature display
    let recipesProcessed = false;     // Flag to prevent double-processing recipes

    // Open SSE connection
    const eventSource = new EventSource(`/stream?query=${encodeURIComponent(query)}`);

    // Add error event listener to debug connection issues
    eventSource.addEventListener('error', function (e) {
      console.error('SSE connection error:', e);
      const spinnerEl = document.getElementById(spinnerId);
      if (spinnerEl) {
        spinnerEl.innerHTML = '<div style="color:red;">Connection error. Please try again.</div>';
      }
    });

    eventSource.onmessage = (event) => {
      let rawData = event.data;

      console.log("Received event data:", rawData);

      // Always parse JSON data from server
      try {
        // Parse the JSON string
        rawData = JSON.parse(rawData);
        console.log("Successfully parsed JSON data:", rawData);
      } catch (e) {
        console.error("Error parsing JSON data:", e);
        // If we can't parse the data, skip this chunk
        return;
      }

      // Special handling for the [DONE] marker
      if (rawData === "[DONE]") {
        console.log("Received [DONE] marker, message complete");
        pendingFormatting = false; // Message is definitely complete

        // Remove spinner, we'll now show final formatted content
        const spinnerEl = document.getElementById(spinnerId);
        if (spinnerEl) spinnerEl.remove();

        // Process final content
        // Check if we've accumulated JSON accidentally
        if (partialResponse.startsWith('[{') || partialResponse.startsWith('{')) {
          console.log('WARNING: Detected JSON in final response - clearing');
          partialResponse = '';

          // Get the spinner to show a message
          const spinnerEl = document.getElementById(spinnerId);
          if (spinnerEl) {
            spinnerEl.innerHTML = '<div style="color:#333;">Loading recipe content...</div>';
          }

          // Return without closing the stream
          return;
        }

        // If we've already processed recipes, skip this step 
        if (recipesProcessed) {
          console.log("Recipes already processed - skipping duplicate processing");
          eventSource.close();
          return;
        }

        const recipes = splitMultipleRecipes(partialResponse);

        if (recipes.length > 0 && (recipes.length > 1 || isRecipeLike(recipes[0]))) {
          console.log(`Final message check: detected ${recipes.length} recipe(s)`);

          // Create new message element for recipe content
          assistantResponseEl = document.createElement("div");
          assistantResponseEl.classList.add("assistant-message");
          chatWindow.appendChild(assistantResponseEl);

          // Store recipe indices for this message
          const recipeIndices = [];

          // Process each recipe
          recipes.forEach((recipeText, idx) => {
            console.log(`Processing final recipe ${idx + 1}/${recipes.length}`);

            // Create a recipe object
            const recipeData = {
              text: recipeText,
              metadata: extractRecipeMetadata(recipeText)
            };

            // * Log out the recipe's data
            console.log("Recipe data:", recipeData);
            console.table(recipeData);

            // Store the recipe
            const recipeIndex = parsedRecipes.length;
            parsedRecipes.push(recipeData);
            recipeIndices.push(recipeIndex);

            // Create a container for this recipe
            const recipeContainer = document.createElement('div');
            recipeContainer.className = 'recipe-container';
            recipeContainer.setAttribute('data-recipe-index', recipeIndex);

            // Render based on current view mode
            if (currentViewMode === 'card') {
              try {
                console.log(`Final render recipe ${idx + 1} as card view`);
                const card = createRecipeCard(recipeData);
                recipeContainer.appendChild(card);
              } catch (e) {
                console.error(`Error rendering final card for recipe ${idx + 1}:`, e);
                // Clean up escaped newlines
                const formattedText = recipeText.replace(/\\n/g, '\n').replace(/\*\*/g, '');
                recipeContainer.innerHTML = marked.parse(formattedText);
              }
            } else {
              console.log(`Final render recipe ${idx + 1} as text view`);
              // Clean up escaped newlines
              const formattedText = recipeText.replace(/\\n/g, '\n').replace(/\*\*/g, '');
              recipeContainer.innerHTML = marked.parse(formattedText);
            }

            // Add this recipe to the message
            assistantResponseEl.appendChild(recipeContainer);
          });

          // Store the recipe indices as a comma-separated list
          assistantResponseEl.setAttribute('data-recipe-indices', recipeIndices.join(','));
          assistantResponseEl.setAttribute('data-is-recipes', 'true');
          assistantResponseEl.setAttribute('data-recipe-count', recipes.length);

          // Add debug info
          console.log(`[Streaming] Stored ${recipeIndices.length} recipe indices: ${recipeIndices.join(',')}`);
          console.log(`[Streaming] Global parsedRecipes array now has ${parsedRecipes.length} recipes`);
        } else {
          // No recipes detected in final content - show as regular text
          assistantResponseEl = document.createElement("div");
          assistantResponseEl.classList.add("assistant-message");

          // Format the text content
          // First, replace escaped newlines with actual newlines
          let cleanText = partialResponse
            .replace(/\\n/g, '\n')  // Replace \n with actual newlines
            .replace(/\*\*/g, '');  // Remove ** formatting

          const markdownDiv = document.createElement('div');
          markdownDiv.className = 'markdown-text';
          markdownDiv.innerHTML = marked.parse(cleanText);

          // Add the content and view mode classes
          assistantResponseEl.appendChild(markdownDiv);
          assistantResponseEl.classList.add(currentViewMode === 'card' ? 'card-view' : 'text-view');

          // Add to DOM
          chatWindow.appendChild(assistantResponseEl);
        }

        // Close the event stream
        eventSource.close();
        return;
      }

      // On first chunk, just mark first chunk as processed - don't create elements yet
      if (isFirstChunk) {
        isFirstChunk = false;
        // We ONLY accumulate content without creating any DOM elements
        // The spinner will remain visible until we're ready to show fully formatted content
      }

      // Special message handling - don't process these
      if (rawData === "[keepalive]") {
        console.log("Keepalive received, ignoring");
        return;
      }

      // Handle spinner marker
      if (rawData === "[spinner]") {
        console.log("Spinner marker received - ensuring spinner is visible");
        // Make sure spinner is visible and any previous message is hidden
        const existingSpinner = document.getElementById(spinnerId);
        if (!existingSpinner) {
          // Recreate spinner if needed
          const newSpinner = document.createElement("div");
          newSpinner.id = spinnerId;
          newSpinner.className = "assistant-message";
          newSpinner.innerHTML = `<div class="spinner"></div>`;
          chatWindow.appendChild(newSpinner);
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        return;
      }

      // Process actual content
      let chunk = rawData;

      // Check if chunk is a JSON array or object
      if ((typeof chunk === 'string' && (chunk.startsWith('[') || chunk.startsWith('{'))) ||
        Array.isArray(chunk) || (typeof chunk === 'object' && chunk !== null)) {
        // If it's already a JSON object or array, don't add it to the partial response
        console.log("Detected raw JSON data - won't display until text is received");

        // If we get raw JSON data, let's wait for the text version
        return;
      }

      // Remove any ** from the chunk before processing
      const cleanChunk = chunk.replace(/\*\*/g, '');
      partialResponse += cleanChunk;

      // Log the content - useful for debugging
      console.log("Received chunk:", chunk.substring(0, 50) + "...");

      // We need to be very careful with our content - always wait for [DONE] marker
      // This ensures we have complete data and never show partial content
      // The only time we display early is for regular text that's not JSON

      // Check if the chunk seems to be a JSON object or contains special markers 
      if (chunk && typeof chunk === 'string' &&
        (chunk.startsWith("{") || chunk.startsWith("[{") || chunk.includes('"nutrition":') || chunk.includes('"recipe":'))) {
        console.log("Detected JSON in response - won't display until complete");
        // Don't set pendingFormatting = false here - wait for [DONE]

        // Skip this chunk to avoid adding JSON to the text view
        return;
      } else if (partialResponse.length > 200 &&
        !partialResponse.includes('{') &&
        !partialResponse.includes('[{') &&
        pendingFormatting) {
        // Only show non-JSON text after accumulating some content
        pendingFormatting = false;
      }

      // Check the accumulated response
      console.log(`Total response length: ${partialResponse.length}`);
      console.log(`Partial response snippet: ${partialResponse.substring(0, 100)}...`);

      // Check if the partial response starts with JSON (a common issue)
      if (partialResponse.startsWith('[{') || partialResponse.startsWith('{') ||
        partialResponse.includes('{"nutrition":') || partialResponse.includes('{"recipe":')) {
        console.log('WARNING: partialResponse contains raw JSON data - clearing and waiting for text content');
        // Clear the partial response to wait for text content
        partialResponse = '';

        // Get the spinner to show a message
        const spinnerEl = document.getElementById(spinnerId);
        if (spinnerEl) {
          spinnerEl.innerHTML = '<div style="color:#333;">Preparing recipe content...</div>';
        }

        return;
      }

      // Try to detect if the accumulated content contains one or more recipes
      // If it contains multiple recipes, split them and process each separately
      let recipes = [];

      // First check for multiple recipes with "Title:" format
      if (partialResponse.includes('Title:')) {
        // Find all instances of "Title:" (case insensitive)
        // Use a more robust regex that matches Title: at the start or after double newlines
        const titleRegex = /(?:^|\n\s*\n\s*)Title:/gi;
        const titleMatches = [...partialResponse.matchAll(titleRegex)];
        console.log(`Found ${titleMatches.length} recipe titles`);

        if (titleMatches.length > 1) {
          // If multiple titles are found, split the text at each Title: marker
          console.log("Multiple recipes detected by Title: markers");
          const splitPoints = titleMatches.map(match => match.index);

          // Split at each Title: marker
          for (let i = 0; i < splitPoints.length; i++) {
            const start = splitPoints[i];
            const end = (i < splitPoints.length - 1) ? splitPoints[i + 1] : partialResponse.length;
            const recipe = partialResponse.substring(start, end).trim();
            if (recipe.length > 0) {
              recipes.push(recipe);
              console.log(`Split recipe ${i + 1} with length ${recipe.length} characters`);
            }
          }
        } else if (titleMatches.length === 1) {
          // If only one title is found, use splitMultipleRecipes as a fallback
          // This handles the case where we have multiple recipes but only detected the first Title:
          console.log("Single recipe with Title: detected, checking for more with splitMultipleRecipes");
          recipes = splitMultipleRecipes(partialResponse);
        }
      }

      // If we didn't find multiple recipes with Title:, try splitMultipleRecipes
      if (recipes.length === 0) {
        console.log("No Title: markers found, using splitMultipleRecipes function");
        recipes = splitMultipleRecipes(partialResponse);
      }

      console.log(`Final recipe count: ${recipes.length}`);

      // Debug the detected recipes
      recipes.forEach((r, i) => {
        console.log(`Recipe ${i + 1} preview: ${r.substring(0, 50)}...`);
      });

      // Only process as recipe if it actually looks like a recipe
      // Check if we have recipes and can render them properly
      if (recipes.length > 0 && (recipes.length > 1 || isRecipeLike(recipes[0])) && !recipesProcessed) {
        console.log(`Processing ${recipes.length} recipe(s)`);

        // Now is the time to create and show the properly formatted content
        pendingFormatting = false;
        recipesProcessed = true; // Set flag to prevent reprocessing in [DONE] handler

        // Remove the spinner first
        const spinnerEl = document.getElementById(spinnerId);
        if (spinnerEl) spinnerEl.remove();

        // Create the message container that will hold the formatted content
        assistantResponseEl = document.createElement("div");
        assistantResponseEl.classList.add("assistant-message");
        chatWindow.appendChild(assistantResponseEl);

        // Store recipe indices for this message
        const recipeIndices = [];

        // Process each recipe
        recipes.forEach((recipeText, idx) => {
          console.log(`Processing recipe ${idx + 1}/${recipes.length}`);

          // Clean the recipe text of any ** characters
          const cleanRecipeText = recipeText.replace(/\*\*/g, '');

          // Create a recipe object
          const recipeData = {
            text: cleanRecipeText,
            metadata: extractRecipeMetadata(cleanRecipeText)
          };

          // Store the recipe
          const recipeIndex = parsedRecipes.length;
          parsedRecipes.push(recipeData);
          recipeIndices.push(recipeIndex);
          console.log(`Stored recipe at index ${recipeIndex}`);

          // Create a container for this recipe
          const recipeContainer = document.createElement('div');
          recipeContainer.className = 'recipe-container';
          recipeContainer.setAttribute('data-recipe-index', recipeIndex);

          // Render based on current view mode
          if (window.currentViewMode === 'card') {
            console.log(`Rendering recipe ${idx + 1} as card view`);
            try {
              const card = createRecipeCard(recipeData);
              recipeContainer.appendChild(card);
            } catch (e) {
              console.error(`Error rendering card for recipe ${idx + 1}:`, e);
              // Clean up escaped newlines first
              const formattedText = cleanRecipeText.replace(/\\n/g, '\n');

              // Use cleaned text for markdown rendering
              recipeContainer.innerHTML = marked.parse(formattedText);
            }
          } else {
            console.log(`Rendering recipe ${idx + 1} as text view`);
            // Clean up escaped newlines first
            const formattedText = cleanRecipeText.replace(/\\n/g, '\n');

            // Use cleaned text for markdown rendering
            recipeContainer.innerHTML = marked.parse(formattedText);
          }

          // Add this recipe to the message
          assistantResponseEl.appendChild(recipeContainer);
        });

        // Store the recipe indices as a comma-separated list
        assistantResponseEl.setAttribute('data-recipe-indices', recipeIndices.join(','));
        assistantResponseEl.setAttribute('data-is-recipes', 'true');
        assistantResponseEl.setAttribute('data-recipe-count', recipes.length);

        // Debug info - make sure all recipes were detected and stored
        console.log(`IMPORTANT: Stored ${recipeIndices.length} recipes in data-recipe-indices: ${recipeIndices.join(',')}`);
        console.log(`IMPORTANT: Set data-recipe-count to ${recipes.length}`);
        console.log(`IMPORTANT: Current view mode when storing recipes: ${window.currentViewMode}`);
      } else {
        // Not a recipe - check if we have enough content to show
        console.log("Not detected as a recipe - checking content length");

        // Only show content if we have at least 250 characters or message seems complete
        const isMessageMeaningful = partialResponse.length > 250 ||
          partialResponse.endsWith('.') ||
          partialResponse.includes('\n\n');

        // Once we have enough meaningful content AND streaming has slowed down, create the element
        if (isMessageMeaningful && !pendingFormatting) {
          // This is a non-recipe message that's ready to be shown

          // Remove the spinner
          const spinnerEl = document.getElementById(spinnerId);
          if (spinnerEl) spinnerEl.remove();

          // Only now create and populate the message element
          assistantResponseEl = document.createElement("div");
          assistantResponseEl.classList.add("assistant-message");

          // Clean the text of any ** characters before rendering
          const cleanText = partialResponse
            .replace(/\\n/g, '\n')  // Replace escaped newlines with actual newlines
            .replace(/\*\*/g, '');  // Remove ** formatting

          // Create the formatted content
          const markdownDiv = document.createElement('div');
          markdownDiv.className = 'markdown-text';
          markdownDiv.innerHTML = marked.parse(cleanText);

          // Add the content and view mode classes
          assistantResponseEl.appendChild(markdownDiv);
          assistantResponseEl.classList.add(window.currentViewMode === 'card' ? 'card-view' : 'text-view');

          // Finally add to DOM when fully ready
          chatWindow.appendChild(assistantResponseEl);
        } else if (partialResponse.length > 400) {
          // If we have a very large response but no recipe detection, force display
          pendingFormatting = false;
        }
      }

      // Auto-scroll
      chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    eventSource.onerror = () => {
      // Handle error
      const spinnerEl = document.getElementById(spinnerId);
      if (spinnerEl) spinnerEl.remove();

      chatWindow.innerHTML += `
        <div class="assistant-message">
          Error occurred while streaming data.
        </div>
      `;

      eventSource.close();
    };
  }
}); // End of DOMContentLoaded event listener

// Function to check if text looks like a recipe
function isRecipeLike(text) {
  if (!text || typeof text !== 'string') return false;

  // First replace escaped newlines with actual newlines
  text = text.replace(/\\n/g, '\n');

  // Log for debugging
  console.log('Checking if text looks like a recipe:', text.substring(0, 100) + '...');

  // Simple, reliable recipe detection - look for the Title: format from our system prompt
  const hasStandardFormat = text.includes('Title:') &&
    (text.includes('Ingredients:') ||
      text.includes('Instructions:'));

  if (hasStandardFormat) {
    console.log('Recipe in standard format detected');
    return true;
  }

  // Fallback detection for markdown formatted recipes 
  const hasMarkdownFormat = /#+\s*(?:Recipe|Ingredients|Instructions|Directions|Method)/i.test(text);
  if (hasMarkdownFormat && text.includes('Ingredients') && text.includes('Instructions')) {
    console.log('Recipe in markdown format detected');
    return true;
  }

  console.log('Not identified as a recipe');
  return false;
}

// Helper function to detect if text is a simple list rather than a recipe
function isSimpleList(text) {
  if (!text) return false;

  // First replace escaped newlines with actual newlines
  text = text.replace(/\\n/g, '\n');

  // Count number of lines that start with numbers or bullet points
  const lines = text.split('\n');
  let listItemCount = 0;
  let totalLines = 0;
  let hasListTitle = false;

  // Check if it starts with list indicators
  if (/^(?:here are|here's|check out|try these|top \d+)\b/i.test(text.trim())) {
    hasListTitle = true;
  }

  // Count list-like items
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length > 0) {
      totalLines++;

      // Look for numbered items or bullet points
      if (/^(?:\d+[\.\)]\s+|\-\s+|\*\s+)/.test(trimmed)) {
        listItemCount++;
      }
    }
  }

  // If more than 50% of non-empty lines are list items, and there's no recipe indicators,
  // it's probably just a list and not a recipe
  const listRatio = listItemCount / totalLines;
  const isListLike = (listRatio > 0.4 && hasListTitle) || listRatio > 0.7;

  // Check for absence of recipe indicators
  const hasNoRecipeIndicators = !text.includes('Ingredients:') &&
    !text.includes('Instructions:') &&
    !text.includes('Directions:') &&
    !text.includes('Method:');

  return isListLike && hasNoRecipeIndicators;
}

// Function to split text into multiple recipes if present
function splitMultipleRecipes(text) {
  if (!text || typeof text !== 'string') return [text];

  // First replace escaped newlines with actual newlines
  text = text.replace(/\\n/g, '\n');

  // Log for debugging
  console.log('Checking for multiple recipes in text');

  // Try to identify recipe boundaries using Title: markers from LLM formatting
  // Use a more robust regex that explicitly looks for Title: after blank lines OR at the beginning of text
  const titleRegex = /(?:^|\n\s*\n\s*)Title:/gi;
  const titleMatches = [...text.matchAll(titleRegex)];

  // If we have multiple Title: markers, split on those
  if (titleMatches.length > 1) {
    console.log(`Found ${titleMatches.length} recipes with Title: markers`);

    // Get the indices of all Title: matches
    const splitIndices = titleMatches.map(match => match.index);
    console.log('Recipe split points:', splitIndices);

    // Split the text at these indices
    const recipes = [];

    for (let i = 0; i < splitIndices.length; i++) {
      const start = splitIndices[i];
      const end = (i < splitIndices.length - 1) ? splitIndices[i + 1] : text.length;

      const recipe = text.substring(start, end).trim();
      if (recipe.length > 0) {
        console.log(`Adding recipe ${i + 1} with length ${recipe.length}`);
        console.log(`Recipe ${i + 1} preview: ${recipe.substring(0, 50)}...`);
        recipes.push(recipe);
      }
    }

    console.log(`Split text into ${recipes.length} recipes with Title: markers`);
    return recipes.length > 0 ? recipes : [text];
  }

  // Special case: Even if only one Title: marker is found, check if the content has multiple
  // recipe patterns like "Title: X" followed by another recipe without a proper Title: marker
  if (titleMatches.length === 1 && text.includes('\n\nTitle:')) {
    // Look for patterns that suggest multiple recipes, like double newlines followed by Recipe Type:
    const recipeTypeMatches = [...text.matchAll(/\n\s*\n\s*(?:Recipe Type:|Cuisine:|Ingredients:)/gi)];

    if (recipeTypeMatches.length > 1) {
      console.log(`Found potential recipe boundaries using secondary markers: ${recipeTypeMatches.length}`);

      // Try to split at these boundaries
      const splitIndices = [titleMatches[0].index]; // Start with the Title: marker

      for (const match of recipeTypeMatches) {
        // Only add this as a split point if it's preceded by a double newline and not too close to the Title:
        const matchIndex = match.index;

        // Check if there's a double newline before this match
        const previousText = text.substring(Math.max(0, matchIndex - 50), matchIndex);
        if (previousText.includes('\n\n') && matchIndex > titleMatches[0].index + 200) {
          // Look backwards from this match to find the double newline
          const precedingText = text.substring(0, matchIndex);
          const lastDoubleNewline = precedingText.lastIndexOf('\n\n');

          if (lastDoubleNewline !== -1 && lastDoubleNewline > splitIndices[splitIndices.length - 1] + 100) {
            splitIndices.push(lastDoubleNewline + 2); // +2 to skip the newlines
          }
        }
      }

      // If we found additional split points, use them
      if (splitIndices.length > 1) {
        console.log('Potential recipe split points:', splitIndices);

        // Split the text at these indices
        const recipes = [];

        for (let i = 0; i < splitIndices.length; i++) {
          const start = splitIndices[i];
          const end = (i < splitIndices.length - 1) ? splitIndices[i + 1] : text.length;

          const recipe = text.substring(start, end).trim();
          if (recipe.length > 0) {
            console.log(`Adding split recipe ${i + 1} with length ${recipe.length}`);
            console.log(`Split recipe ${i + 1} preview: ${recipe.substring(0, 50)}...`);
            recipes.push(recipe);
          }
        }

        if (recipes.length > 1) {
          console.log(`Split text into ${recipes.length} recipes with combined markers`);
          return recipes;
        }
      }
    }
  }

  // Fallback: check for markdown headings with recipe titles
  const headingMatches = [...text.matchAll(/(?:^|\n\s*\n\s*)#+\s+(.+?)(?:\n|$)/g)];

  if (headingMatches.length > 1) {
    console.log(`Found ${headingMatches.length} potential recipe headings`);

    // Get the indices of all heading matches
    const splitIndices = headingMatches.map(match => match.index);
    console.log('Recipe split points with markdown headings:', splitIndices);

    // Split the text at these indices
    const recipes = [];

    for (let i = 0; i < splitIndices.length; i++) {
      const start = splitIndices[i];
      const end = (i < splitIndices.length - 1) ? splitIndices[i + 1] : text.length;

      const recipe = text.substring(start, end).trim();
      console.log(`Checking potential recipe ${i + 1}: ${recipe.substring(0, 40)}...`);

      // Only add if it looks like a recipe
      if (recipe.length > 0 && isRecipeLike(recipe)) {
        console.log(`Adding recipe ${i + 1} with heading`);
        recipes.push(recipe);
      }
    }

    if (recipes.length > 1) {
      console.log(`Split text into ${recipes.length} markdown-formatted recipes`);
      return recipes;
    }
  }

  // Another fallback: Try to detect double newlines + ingredient lists
  const ingredientSections = [...text.matchAll(/(?:\n\s*\n\s*)(?:Ingredients?:|ingredients?:)/gi)];
  if (ingredientSections.length > 1) {
    console.log(`Found ${ingredientSections.length} ingredient sections, might be multiple recipes`);

    // Get the indices of all ingredient section starts
    let sectionStarts = [];
    let previousEnd = 0;

    for (const match of ingredientSections) {
      // Find the start of the recipe containing this ingredient section
      // Look backward from the ingredient section to find the previous section break
      const precedingText = text.substring(previousEnd, match.index);
      const lastDoubleNewline = precedingText.lastIndexOf('\n\n');

      if (lastDoubleNewline !== -1) {
        sectionStarts.push(previousEnd + lastDoubleNewline + 2); // +2 to skip the newlines
      } else {
        sectionStarts.push(match.index);
      }

      previousEnd = match.index;
    }

    console.log('Recipe section starts:', sectionStarts);

    // Split the text at these section starts
    const recipes = [];

    for (let i = 0; i < sectionStarts.length; i++) {
      const start = sectionStarts[i];
      const end = (i < sectionStarts.length - 1) ? sectionStarts[i + 1] : text.length;

      const recipe = text.substring(start, end).trim();
      if (recipe.length > 0) {
        console.log(`Adding recipe section ${i + 1} with length ${recipe.length}`);
        console.log(`Recipe section ${i + 1} preview: ${recipe.substring(0, 50)}...`);
        recipes.push(recipe);
      }
    }

    if (recipes.length > 1) {
      console.log(`Split text into ${recipes.length} recipes by ingredient sections`);
      return recipes;
    }
  }

  console.log('No multiple recipes detected, returning as single recipe');
  return [text];
}

// Function to extract recipe metadata from text
function extractRecipeMetadata(text) {
  console.log('Extracting metadata from recipe text');

  const metadata = {
    recipe_title: 'Recipe',
    source: 'ChefBoost AI', // Default source
    date_issued: new Date().toLocaleDateString() // Default date
  };

  // Clean up text by removing any ** characters and handling escaped newlines
  text = text.replace(/\*\*/g, '').replace(/\\n/g, '\n');

  // Try to extract source information from text - match entire line to capture multi-word sources
  const sourceMatch = text.match(/Source:(.+?)(?:\n|$)/i);
  if (sourceMatch && sourceMatch[1].trim()) {
    metadata.source = sourceMatch[1].trim();
    // If source is just "ChefBoost AI" and there's a Source section in text, look for possible book reference
    if (metadata.source === "ChefBoost AI" || metadata.source === "Generated by ChefBoost AI") {
      // Try to find a book reference in the text
      const bookMatch = text.match(/(?:from|in)\s+(?:the\s+)?(?:book|cookbook)?\s*["']?([^"'\n.]+)["']?/i);
      if (bookMatch && bookMatch[1].trim()) {
        metadata.source = bookMatch[1].trim();
      }
    }
  }

  // Look for date information
  const dateMatch = text.match(/Date:(.+?)(?:\n|$)/i);
  if (dateMatch && dateMatch[1].trim()) {
    metadata.date_issued = dateMatch[1].trim();
  }

  // Method 1: Look for explicit Title field
  const titleMatch = text.match(/Title:\s*([^\n]+)/i);
  if (titleMatch && titleMatch[1]) {
    metadata.recipe_title = titleMatch[1].trim();
    console.log('Found title using Title: pattern:', metadata.recipe_title);
  } else {
    // Method 2: Look for markdown headings as title
    const headingMatch = text.match(/^#+ (.+)$/m);
    if (headingMatch && headingMatch[1]) {
      metadata.recipe_title = headingMatch[1].trim();
      console.log('Found title using markdown heading:', metadata.recipe_title);
    } else {
      // Method 3: First line might be the title if short
      const lines = text.split('\n');
      if (lines[0] && lines[0].length < 60 && !lines[0].includes(':')) {
        metadata.recipe_title = lines[0].trim();
        console.log('Using first line as title:', metadata.recipe_title);
      }
    }
  }

  // Directly look for Recipe Type: field first (most reliable)
  const typeMatch = text.match(/Recipe Type:\s*([^\n,]+)/i);
  if (typeMatch && typeMatch[1]) {
    metadata.recipe_type = typeMatch[1].trim().toLowerCase();
    console.log('Found explicit recipe type:', metadata.recipe_type);
  } else {
    // Fallback: Try to extract type by keywords
    const recipeTypes = {
      'dessert': ['dessert', 'cake', 'cookie', 'pie', 'sweet', 'chocolate', 'pudding', 'ice cream'],
      'soup': ['soup', 'stew', 'broth', 'chowder'],
      'salad': ['salad', 'slaw'],
      'appetizer': ['appetizer', 'starter', 'snack', 'dip', 'finger food'],
      'beverage': ['drink', 'beverage', 'cocktail', 'smoothie', 'juice'],
      'breakfast': ['breakfast', 'morning', 'eggs', 'pancake', 'waffle'],
      'main course': ['dinner', 'lunch', 'entree', 'main dish', 'main course']
    };

    // Default to main course
    metadata.recipe_type = 'main course';

    // Check for type keywords
    Object.entries(recipeTypes).forEach(([type, keywords]) => {
      for (const keyword of keywords) {
        if (text.toLowerCase().includes(keyword)) {
          metadata.recipe_type = type;
          console.log(`Detected recipe type: ${type} (from keyword: ${keyword})`);
          break;
        }
      }
    });
  }

  // Directly look for Cuisine: field first (most reliable)
  const cuisineMatch = text.match(/Cuisine:\s*([^\n,]+)/i);
  if (cuisineMatch && cuisineMatch[1]) {
    metadata.cuisine = cuisineMatch[1].trim().toLowerCase();
    console.log('Found explicit cuisine:', metadata.cuisine);
  } else {
    // Fallback: Try to identify cuisine by keywords
    const cuisines = [
      'italian', 'french', 'thai', 'japanese', 'chinese', 'mexican', 'indian',
      'greek', 'spanish', 'german', 'american', 'british', 'irish', 'moroccan',
      'turkish', 'lebanese', 'vietnamese', 'korean', 'cajun', 'creole', 'southern'
    ];

    for (const cuisine of cuisines) {
      if (text.toLowerCase().includes(cuisine)) {
        metadata.cuisine = cuisine;
        console.log('Detected cuisine:', cuisine);
        break;
      }
    }
  }

  // Directly look for Special Considerations: field first (most reliable)
  const considerationsMatch = text.match(/Special Considerations:\s*([^\n]+)/i);
  if (considerationsMatch && considerationsMatch[1]) {
    // Use the exact special considerations
    metadata.special_considerations = considerationsMatch[1].trim();
    console.log('Found explicit considerations:', metadata.special_considerations);
  } else {
    // Fallback: Try to identify dietary info by keywords
    const dietary = [
      'vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'nut-free',
      'low-carb', 'keto', 'paleo', 'low-fat', 'sugar-free', 'egg-free'
    ];

    const considerations = [];
    for (const diet of dietary) {
      if (text.toLowerCase().includes(diet)) {
        considerations.push(diet);
        console.log('Detected dietary consideration:', diet);
      }
    }

    if (considerations.length > 0) {
      metadata.special_considerations = considerations.join(', ');
    }
  }

  // Try to extract ingredients
  metadata.ingredients = extractIngredients(text);

  console.log('Final extracted metadata:', metadata);
  return metadata;
}

// Helper function to extract ingredients from text
function extractIngredients(text) {
  const ingredients = [];

  // First replace escaped newlines with actual newlines
  text = text.replace(/\\n/g, '\n');

  // Then try to find an ingredients section
  const ingredientsSection = extractSection(text, 'Ingredients');

  if (ingredientsSection) {
    // Split by lines and look for bullet points or numbered items
    const lines = ingredientsSection.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      // Look for bullet points, numbers, or dashes at start of line
      if (/^[\*\-•]|^\d+\.?\s+/.test(trimmed) && trimmed.length > 2) {
        // Remove the bullet/number and add to ingredients
        const ingredient = trimmed.replace(/^[\*\-•]|^\d+\.?\s+/, '').trim();
        if (ingredient) {
          ingredients.push(ingredient);
        }
      }
    }
  } else {
    // If no ingredients section, look for bullet points throughout the text
    const lines = text.split('\n');

    // Check if we're in an ingredient-like section
    let inIngredientsSection = false;

    for (const line of lines) {
      const trimmed = line.trim();

      // Start of ingredients section markers
      if (/^#+\s+ingredients/i.test(trimmed) || /^ingredients/i.test(trimmed)) {
        inIngredientsSection = true;
        continue;
      }

      // End of ingredients section markers (start of another section)
      if (inIngredientsSection && /^#+\s+/i.test(trimmed) && !/ingredient/i.test(trimmed)) {
        inIngredientsSection = false;
        continue;
      }

      // Process lines in ingredients section
      if (inIngredientsSection && trimmed.length > 2) {
        // Look for bullet points or numbers
        if (/^[\*\-•]|^\d+\.?\s+/.test(trimmed)) {
          const ingredient = trimmed.replace(/^[\*\-•]|^\d+\.?\s+/, '').trim();
          if (ingredient) {
            ingredients.push(ingredient);
          }
        }
      }
    }
  }

  return ingredients.length > 0 ? ingredients : [];
}

// TODO: Define toggleView on the window object to ensure it's accessible globally

// TODO: Function to switch between views

// * Get the chat window 
const chatWindow = document.getElementById('chat-window');
if (!chatWindow) {
  console.error('Chat window not found!');
  return;
}

console.log('Found chat window, applying view classes to all messages');
const allMessages = chatWindow.querySelectorAll('.assistant-message');

// Apply view classes to all messages first
allMessages.forEach(msg => {
  // Add the view mode class to the message
  msg.classList.remove('card-view', 'text-view');
  msg.classList.add(mode === 'card' ? 'card-view' : 'text-view');

  // IMPORTANT: For non-recipe content in card view, make sure it's visible
  // Check if this is a non-recipe message
  const isRecipeMessage =
    msg.hasAttribute('data-recipe-index') ||
    msg.hasAttribute('data-recipe-indices') ||
    msg.hasAttribute('data-is-recipe') ||
    msg.hasAttribute('data-is-recipes');

  // If we're in card view and this is NOT a recipe message, make sure text is visible
  if (mode === 'card' && !isRecipeMessage) {
    const textContent = msg.querySelector('.markdown-text');
    if (textContent) {
      textContent.style.display = 'block';
      console.log('Ensuring non-recipe content is visible in card view');
    }
  }
});

// Now force rerendering of all messages to ensure proper content display
if (mode === 'card') {
  console.log('Forcing rerender of all recipe messages in card view format');
  forceRenderAllRecipeMessages();
} else {
  console.log('Forcing rerender of all recipe messages in text view format');
  forceRenderAllTextMessages();
}

// Rerender assistant messages with the new view
console.log('Found chat window, searching for messages for content updates');
const assistantMessages = chatWindow.querySelectorAll('.assistant-message');
console.log(`Found ${assistantMessages.length} assistant messages to check for recipes`);

let recipeCount = 0;

// Helper to render all recipe messages as cards
function forceRenderAllRecipeMessages() {
  allMessages.forEach((msg, i) => {
    // Check if message contains any recipe attributes 
    const hasRecipeAttributes = msg.hasAttribute('data-recipe-index') ||
      msg.hasAttribute('data-recipe-indices') ||
      msg.hasAttribute('data-is-recipe') ||
      msg.hasAttribute('data-is-recipes');

    // Only if it's already marked as a recipe, show the card view                      
    if (hasRecipeAttributes) {
      console.log(`Forcing card render for message ${i}`);

      // Clear the existing message content and rebuild it
      if (msg.hasAttribute('data-recipe-indices')) {
        // Get indices and rebuild all recipes in card view
        const indicesStr = msg.getAttribute('data-recipe-indices');
        console.log(`***DEBUG: Raw indices string: "${indicesStr}"`);

        const indices = indicesStr.split(',').map(i => parseInt(i, 10));
        console.log(`Message ${i} has recipe indices: ${indices}`);

        // Create new content
        const validIndices = indices.filter(idx => {
          const isValid = parsedRecipes[idx] !== undefined;
          console.log(`Index ${idx} is ${isValid ? 'valid' : 'invalid'}`);
          return isValid;
        });

        if (validIndices.length > 0) {
          console.log(`Building card view for ${validIndices.length} recipes`);

          // First, save the existing indices
          const existingIndices = msg.getAttribute('data-recipe-indices');
          const recipeCount = msg.getAttribute('data-recipe-count') || validIndices.length;

          console.log(`***CARD DEBUG: Message has data-recipe-indices: ${existingIndices}`);
          console.log(`***CARD DEBUG: Message has data-recipe-count: ${recipeCount}`);

          // Clear and rebuild
          msg.innerHTML = '';

          // Re-add the attributes
          msg.setAttribute('data-recipe-indices', existingIndices);
          msg.setAttribute('data-is-recipes', 'true');
          msg.setAttribute('data-recipe-count', recipeCount);
          msg.classList.add('card-view');
          msg.classList.remove('text-view');

          // Rebuild all recipe containers
          validIndices.forEach((recipeIndex, idx) => {
            const recipe = parsedRecipes[recipeIndex];
            if (recipe) {
              console.log(`***CARD DEBUG: Processing recipe ${idx + 1}/${validIndices.length} with index ${recipeIndex}`);
              console.log(`***CARD DEBUG: Recipe title: ${recipe.metadata?.recipe_title || 'Unknown'}`);

              // Create a container for this recipe
              const recipeContainer = document.createElement('div');
              recipeContainer.className = 'recipe-container';
              recipeContainer.setAttribute('data-recipe-index', recipeIndex);

              try {
                const card = createRecipeCard(recipe);
                recipeContainer.appendChild(card);
              } catch (e) {
                console.error(`Error creating card for recipe ${idx}:`, e);
                const formattedText = recipe.text.replace(/\\n/g, '\n').replace(/\*\*/g, '');
                recipeContainer.innerHTML = marked.parse(formattedText);
              }

              msg.appendChild(recipeContainer);
              console.log(`Added recipe ${idx + 1}/${validIndices.length} to card view`);
            }
          });
        }
      } else {
        // Handle single recipe case
        const recipeIndex = parseInt(msg.getAttribute('data-recipe-index') || -1);
        if (recipeIndex >= 0 && parsedRecipes[recipeIndex]) {
          // Hide any existing text content
          const textContent = msg.querySelector('.markdown-text');
          if (textContent) {
            textContent.style.display = 'none';
          }

          // Make sure we're in card view mode
          msg.classList.add('card-view');
          msg.classList.remove('text-view');

          // If there's no card yet, create one
          if (!msg.querySelector('.recipe-card')) {
            // Create a new card
            const recipe = parsedRecipes[recipeIndex];
            try {
              const card = createRecipeCard(recipe);
              msg.innerHTML = '';
              msg.appendChild(card);
            } catch (e) {
              console.error(`Error creating card:`, e);
            }
          } else {
            // Show existing cards
            const recipeCards = msg.querySelectorAll('.recipe-card');
            recipeCards.forEach(card => {
              card.style.display = 'block';
            });
          }
        }
      }
    }
  });
}

// Helper to render all messages as text
function forceRenderAllTextMessages() {
  allMessages.forEach((msg, i) => {
    // Check if message contains any recipe attributes 
    const hasRecipeAttributes = msg.hasAttribute('data-recipe-index') ||
      msg.hasAttribute('data-recipe-indices') ||
      msg.hasAttribute('data-is-recipe') ||
      msg.hasAttribute('data-is-recipes');

    if (hasRecipeAttributes) {
      console.log(`Forcing text render for message ${i}`);

      // Clear the existing message content and rebuild it using text format
      if (msg.hasAttribute('data-recipe-indices')) {
        // Get indices and rebuild all recipes in text view
        const indicesStr = msg.getAttribute('data-recipe-indices');
        console.log(`***TEXT DEBUG: Raw indices string: "${indicesStr}"`);

        const indices = indicesStr.split(',').map(i => parseInt(i, 10));
        console.log(`Message ${i} has recipe indices: ${indices}`);

        // Create new content
        const validIndices = indices.filter(idx => {
          const isValid = parsedRecipes[idx] !== undefined;
          console.log(`Index ${idx} is ${isValid ? 'valid' : 'invalid'} for text view`);
          return isValid;
        });

        if (validIndices.length > 0) {
          console.log(`Building text view for ${validIndices.length} recipes`);

          // First, save the existing indices
          const existingIndices = msg.getAttribute('data-recipe-indices');
          const recipeCount = msg.getAttribute('data-recipe-count') || validIndices.length;

          console.log(`***TEXT DEBUG: Message has data-recipe-indices: ${existingIndices}`);
          console.log(`***TEXT DEBUG: Message has data-recipe-count: ${recipeCount}`);

          // Clear and rebuild
          msg.innerHTML = '';

          // Re-add the attributes
          msg.setAttribute('data-recipe-indices', existingIndices);
          msg.setAttribute('data-is-recipes', 'true');
          msg.setAttribute('data-recipe-count', recipeCount);
          msg.classList.add('text-view');
          msg.classList.remove('card-view');

          // Create a common container if multiple recipes
          const recipesWrapper = document.createElement('div');
          recipesWrapper.className = 'markdown-text';
          msg.appendChild(recipesWrapper);

          // Rebuild all recipe containers
          validIndices.forEach((recipeIndex, idx) => {
            const recipe = parsedRecipes[recipeIndex];
            if (recipe) {
              console.log(`***TEXT DEBUG: Processing recipe ${idx + 1}/${validIndices.length} with index ${recipeIndex}`);
              console.log(`***TEXT DEBUG: Recipe title: ${recipe.metadata?.recipe_title || 'Unknown'}`);

              // Create a container for this recipe
              const recipeContainer = document.createElement('div');
              recipeContainer.className = 'recipe-container';
              recipeContainer.setAttribute('data-recipe-index', recipeIndex);

              try {
                // Make sure we have complete recipe text
                const markdown = formatRecipeAsMarkdown(recipe);
                const formattedText = markdown.replace(/\\n/g, '\n');
                console.log(`***TEXT DEBUG: Recipe text snippet: ${formattedText.substring(0, 100)}...`);
                recipeContainer.innerHTML = marked.parse(formattedText);
              } catch (e) {
                console.error(`Error formatting markdown for recipe ${idx}:`, e);
                const formattedText = recipe.text.replace(/\\n/g, '\n').replace(/\*\*/g, '');
                recipeContainer.innerHTML = marked.parse(formattedText);
              }

              // Add a divider between recipes except for the last one
              if (idx < validIndices.length - 1) {
                const divider = document.createElement('hr');
                divider.style.margin = '20px 0';
                recipeContainer.appendChild(divider);
              }

              recipesWrapper.appendChild(recipeContainer);
              console.log(`Added recipe ${idx + 1}/${validIndices.length} to text view`);
            }
          });
        }
      } else {
        // Handle single recipe case
        const recipeIndex = parseInt(msg.getAttribute('data-recipe-index') || -1);
        if (recipeIndex >= 0 && parsedRecipes[recipeIndex]) {
          // Make sure we're in text view mode
          msg.classList.add('text-view');
          msg.classList.remove('card-view');

          // Hide any recipe cards
          const recipeCards = msg.querySelectorAll('.recipe-card');
          recipeCards.forEach(card => {
            card.style.display = 'none';
          });

          // If there's no text content yet, create it
          if (!msg.querySelector('.markdown-text')) {
            // Create text content
            const recipe = parsedRecipes[recipeIndex];
            try {
              const markdown = formatRecipeAsMarkdown(recipe);
              const formattedText = markdown.replace(/\\n/g, '\n');

              // Create a wrapper for the markdown content
              const markdownWrapper = document.createElement('div');
              markdownWrapper.className = 'markdown-text';
              markdownWrapper.innerHTML = marked.parse(formattedText);

              // Clear existing content and add the wrapper
              msg.innerHTML = '';
              msg.appendChild(markdownWrapper);
            } catch (e) {
              console.error(`Error creating text view:`, e);
            }
          } else {
            // Show existing text content
            const textContent = msg.querySelector('.markdown-text');
            if (textContent) {
              textContent.style.display = 'block';
            }
          }
        }
      }
    } else {
      // Not a recipe message - just show any text content
      const textContent = msg.querySelector('.markdown-text');
      if (textContent) {
        textContent.style.display = 'block';
      }

      // Hide any recipe cards
      const recipeCards = msg.querySelectorAll('.recipe-card');
      recipeCards.forEach(card => {
        card.style.display = 'none';
      });
    }
  });
}

assistantMessages.forEach((msg, index) => {
  // Skip any message that has the spinner
  if (msg.querySelector('.spinner')) {
    console.log(`Message ${index} contains spinner, skipping`);
    return;
  }

  // Check if this message contains multiple recipes
  if (msg.hasAttribute('data-recipe-indices')) {
    // Get the recipe indices for this message
    const recipeIndicesStr = msg.getAttribute('data-recipe-indices');
    let recipeIndices = recipeIndicesStr.split(',').map(idx => parseInt(idx));

    console.log(`Message ${index} has recipes with indices: ${recipeIndicesStr}`);

    if (recipeIndices.length > 0) {
      // Check for duplicate recipe indices (a common issue)
      const uniqueIndices = [...new Set(recipeIndices)];
      if (uniqueIndices.length < recipeIndices.length) {
        console.log(`Detected duplicate recipe indices - using unique indices instead of ${recipeIndices}`);
        recipeIndices = uniqueIndices;
      }

      // Filter to just valid indices
      const validIndices = recipeIndices.filter(idx => parsedRecipes[idx]);
      if (validIndices.length < recipeIndices.length) {
        console.log(`Some recipe indices were invalid, using only valid ones: ${validIndices}`);
        recipeIndices = validIndices;
      }

      // Store recipe count for reference
      const recipeCount = validIndices.length;
      msg.setAttribute('data-recipe-count', recipeCount);

      // Based on mode, call appropriate render function
      if (mode === 'card') {
        console.log(`Using card view mode for message ${index} with ${recipeCount} recipes`);
        // We don't need to do anything here - forceRenderAllRecipeMessages will handle it
      } else {
        console.log(`Using text view mode for message ${index} with ${recipeCount} recipes`);
        // We don't need to do anything here - forceRenderAllTextMessages will handle it
      }
    }
  }
  // Check for single recipe (legacy format)
  else {
    const recipeIndex = parseInt(msg.getAttribute('data-recipe-index') || -1);
    console.log(`Message ${index} has single recipe index: ${recipeIndex}`);

    if (recipeIndex >= 0 && parsedRecipes[recipeIndex]) {
      recipeCount++;
      console.log(`Found recipe data for message ${index}`);

      // Based on mode, set appropriate class
      if (mode === 'card') {
        msg.classList.add('card-view');
        msg.classList.remove('text-view');
      } else {
        msg.classList.add('text-view');
        msg.classList.remove('card-view');
      }
    } else {
      console.log(`No recipe data found for message ${index}`);
      // For messages without recipe data, mark them with a class
      msg.classList.add('non-recipe-message');
    }
  }
});

console.log(`Processed ${recipeCount} recipe messages out of ${assistantMessages.length} total messages`);

  // No test cards in production
}

// TODO: Create recipe card from template
function createRecipeCard(recipeData) {
  // TODO: Get template

  // TODO: Clone the template content

  // Add margins to all text content elements to prevent text overflow
  const textElements = card.querySelectorAll('p, li, h3, h4');
  textElements.forEach(el => {
    el.style.margin = "8px";
    el.style.wordWrap = "break-word";
    el.style.overflowWrap = "break-word";
    el.style.padding = "0 5px";
    el.style.boxSizing = "border-box";
    el.style.maxWidth = "100%";
  });

  // TODO: Update the recipe card template
  try {
    // TODO: Extract recipe information

    // Clean up title by removing numbering prefixes and any **
    let title = recipe.metadata?.recipe_title || 'Recipe';
    title = title.replace(/^\d+\.\s*/, ''); // Remove any leading numbers like "2. "
    title = title.replace(/\*\*/g, ''); // Remove any ** from title

    // Populate recipe header
    const titleEl = card.querySelector('.recipe-title');
    const typeEl = card.querySelector('.recipe-type');
    const cuisineEl = card.querySelector('.recipe-cuisine');
    const considerationsEl = card.querySelector('.recipe-special-considerations');

    if (titleEl) titleEl.textContent = title;

    if (typeEl && recipe.metadata?.recipe_type) {
      let type = recipe.metadata.recipe_type;
      // Capitalize first letter of type
      type = type.charAt(0).toUpperCase() + type.slice(1);
      // Remove any ** from type
      type = type.replace(/\*\*/g, '');
      typeEl.textContent = type;
    } else if (typeEl) {
      typeEl.textContent = '';
    }

    if (cuisineEl && recipe.metadata?.cuisine) {
      let cuisine = recipe.metadata.cuisine;
      // Capitalize first letter of cuisine
      cuisine = cuisine.charAt(0).toUpperCase() + cuisine.slice(1);
      // Remove any ** from cuisine
      cuisine = cuisine.replace(/\*\*/g, '');
      cuisineEl.textContent = cuisine;
    } else if (cuisineEl) {
      cuisineEl.textContent = '';
    }

    if (considerationsEl) {
      let considerations = recipe.metadata?.special_considerations;
      // Handle different formats: string or array
      if (Array.isArray(considerations)) {
        considerations = considerations.join(', ');
      }
      // Handle or clean up considerations
      if (typeof considerations === 'string') {
        // Remove any ** from considerations
        considerations = considerations.replace(/\*\*/g, '');
        considerationsEl.textContent = considerations;
      } else {
        considerationsEl.textContent = '';
      }
    }

    // TODO: Populate ingredients

    // TODO: Add instructions

    // Clean up instructions before rendering
    // Handle escaped newlines
    instructionsText = instructionsText.replace(/\\n/g, '\n');

    // Remove leading asterisks that might appear from markdown formatting
    instructionsText = instructionsText.replace(/^\s*\*\*\s*$/gm, '');

    // Add some space between paragraphs for better readability
    instructionsText = instructionsText.replace(/\n\n/g, '\n\n\n');

    // Render as markdown
    instructionsContent.innerHTML = marked.parse(instructionsText);

    // Add inline styles to ensure content is properly contained
    instructionsContent.style.padding = "0 10px";
    instructionsContent.style.boxSizing = "border-box";
    instructionsContent.style.width = "100%";
    instructionsContent.style.wordWrap = "break-word";
    instructionsContent.style.overflowWrap = "break-word";
  }

    // TODO: Set up source info panel toggle

    // TODO: Return the card

    // TODO: Catch any errors and print a message in a div within the recipe card

// * This is the closing curly brace for the createRecipeCard function:
}

// Helper function to extract sections from recipe text
function extractSection(text, sectionName) {
  if (!text) return null;

  // First replace escaped newlines with actual newlines
  text = text.replace(/\\n/g, '\n');

  const regex = new RegExp(`${sectionName}:\\s*([\\s\\S]*?)(?=\\n\\n|\\n[A-Z][a-z]+:|$)`, 'i');
  const match = text.match(regex);
  return match ? match[1].trim() : null;
}

// Format recipe as markdown for text view
function formatRecipeAsMarkdown(recipeData) {
  try {
    let recipe = recipeData.recipe || recipeData;

    // If it's already formatted text, just return it after cleaning up double asterisks and escaped newlines
    if (typeof recipeData === 'string') {
      return recipeData.replace(/\*\*/g, '').replace(/\\n/g, '\n');
    }

    // Build clean markdown representation
    let markdown = '';

    // Extract title, removing any numbering prefixes
    let title = recipe.metadata?.recipe_title || 'Recipe';
    title = title.replace(/^\d+\.\s*/, ''); // Remove any leading numbers like "2. "
    title = title.replace(/\*\*/g, ''); // Remove any ** from the title

    // Add title
    markdown += `## ${title}\n\n`;

    // Add metadata in a single row
    if (recipe.metadata) {
      let metadataItems = [];
      if (recipe.metadata.recipe_type) {
        let type = recipe.metadata.recipe_type;
        // Capitalize first letter of type
        type = type.charAt(0).toUpperCase() + type.slice(1);
        type = type.replace(/\*\*/g, ''); // Remove any ** from the type
        metadataItems.push(`Type: ${type}`);
      }

      if (recipe.metadata.cuisine) {
        let cuisine = recipe.metadata.cuisine;
        // Capitalize first letter of cuisine
        cuisine = cuisine.charAt(0).toUpperCase() + cuisine.slice(1);
        cuisine = cuisine.replace(/\*\*/g, ''); // Remove any ** from the cuisine
        metadataItems.push(`Cuisine: ${cuisine}`);
      }

      if (recipe.metadata.special_considerations) {
        const considerations = Array.isArray(recipe.metadata.special_considerations)
          ? recipe.metadata.special_considerations.join(', ')
          : recipe.metadata.special_considerations;

        if (considerations) {
          const cleanConsiderations = considerations.replace(/\*\*/g, '');
          metadataItems.push(`Special Considerations: ${cleanConsiderations}`);
        }
      }

      if (metadataItems.length > 0) {
        markdown += metadataItems.join(' | ') + '\n\n';
      }
    }

    // Process recipe text to avoid repetition and remove **
    if (recipe.text) {
      // First, remove the title if it exists at the start of the text
      let processedText = recipe.text;

      // Remove Title: line if present
      processedText = processedText.replace(/^Title:\s*[^\n]+\n*/i, '');

      // Remove numbered title if present (like "2. Recipe Name")
      processedText = processedText.replace(/^\d+\.\s*[^\n]+\n*/i, '');

      // If the text starts with the recipe title, remove it
      const escapedTitle = title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const titleRegex = new RegExp(`^${escapedTitle}\\s*\\n+`, 'i');
      processedText = processedText.replace(titleRegex, '');

      // Remove all ** from the text
      processedText = processedText.replace(/\*\*/g, '');

      markdown += processedText.trim() + '\n\n';
    }

    // Always add source information
    markdown += '### Source Information\n';

    // Use source info if available, or default to ChefBoost AI
    if (recipe.metadata) {
      // First check for explicit source field, which should contain book title or origin
      const sourceText = recipe.metadata.source || recipe.metadata.title || 'ChefBoost AI';
      let cleanSource = sourceText.replace(/\*\*/g, '');

      // Check if it's just "ChefBoost AI" and try to extract any book references from the text
      if ((cleanSource === 'ChefBoost AI' || cleanSource === 'Generated by ChefBoost AI') && recipe.text) {
        const bookMatch = recipe.text.match(/(?:from|in)\s+(?:the\s+)?(?:book|cookbook)?\s*["']?([^"'\n.]+)["']?/i);
        if (bookMatch && bookMatch[1].trim()) {
          cleanSource = bookMatch[1].trim();
        }
      }

      markdown += `Source: ${cleanSource}\n`;

      if (recipe.metadata.authors) {
        const authors = Array.isArray(recipe.metadata.authors)
          ? recipe.metadata.authors.join(', ')
          : recipe.metadata.authors;
        const cleanAuthors = authors.replace(/\*\*/g, '');
        markdown += `Author(s): ${cleanAuthors}\n`;
      }

      const dateText = recipe.metadata.date_issued || new Date().toLocaleDateString();
      const cleanDate = dateText.toString().replace(/\*\*/g, '');
      markdown += `Date: ${cleanDate}\n`;
    } else {
      markdown += `Source: ChefBoost AI\n`;
      markdown += `Date: ${new Date().toLocaleDateString()}\n`;
    }

    markdown += '\n';

    return markdown;
  } catch (e) {
    console.error('Error formatting recipe as markdown:', e);
    return 'Error displaying recipe information';
  }
}

// Expose the main functions to the window object so they can be called from inline scripts
// Make sure to do this synchronously at the end of file
window.switchView = switchView;
// toggleView is already defined on window
window.createRecipeCard = createRecipeCard;
window.formatRecipeAsMarkdown = formatRecipeAsMarkdown;
// parsedRecipes and currentViewMode are already defined on window
window.marked = marked;

// Ensure these are properly set
console.log('GLOBAL FUNCTIONS EXPOSED:');
console.log('window.switchView =', typeof window.switchView === 'function' ? 'AVAILABLE' : 'MISSING');
console.log('window.createRecipeCard =', typeof window.createRecipeCard === 'function' ? 'AVAILABLE' : 'MISSING');
console.log('window.formatRecipeAsMarkdown =', typeof window.formatRecipeAsMarkdown === 'function' ? 'AVAILABLE' : 'MISSING');