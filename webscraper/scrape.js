//Recipe Scraper using Puppeteer by Jack Byrne

import { launch } from 'puppeteer';
import { mongoose } from 'mongoose';

const mongoString = 'mongodb://127.0.0.1:27017/JBrecipegen'

//manually find and define url and css selectors for the website you want to scrape
const url = 'https://publicdomainrecipes.org';
const ingredientscss = '.ingredients';
const instructionscss = '.directions';
const recipeLinkcss = 'h3 a';
const prepTimecss = 'n/a';
const categoriescss = '.taxo-display:not(.flex-container) a[href*="/categories/"]';
const equipmentcss = '.taxo-display:not(.flex-container) a[href*="/equipment/"]';

const numberOfPages = 120; // Adjust based on how many pages you want to scrape

async function scrapeRecipe(url) {
  // Launch the browser and go to the recipe page  
  const browser = await launch({ headless: "new" });
  const page = await browser.newPage();
  await page.goto(url);

  // Get the data
  const title = await page.$eval('h1', h1 => h1.textContent);
  // Get the ingredients and remove unwanted parts
  const ingredientsRaw = await page.$eval(ingredientscss, div => div.textContent);
  const ingredients = ingredientsRaw
    .split('\n') // Split by newline
    .map(item => item.trim()) // Trim whitespace
    .filter(item => item.length > 0 && item !== 'Ingredients'); // Remove empty lines and the title "Ingredients"

  // Get the instructions and split them into steps
  const instructionsRaw = await page.$eval(instructionscss, div => div.textContent);
  const instructions = instructionsRaw
    .replace('Directions', '') // Remove the title "Directions"
    .split('\n') // Split by newline
    .map(step => step.trim()) // Trim whitespace
    .filter(step => step.length > 0); // Remove empty lines

  const prepTime = await page.$eval(prepTimecss, div => div.textContent.trim()).catch(() => null); // Replace with correct CSS selector
  const categories = await page.$$eval(categoriescss, items => items.map(item => item.textContent)).catch(() => []); // Replace with correct CSS selector
  const equipment = await page.$$eval(equipmentcss, items => items.map(item => item.textContent)).catch(() => []); // Replace with correct CSS selector


 // Close the browser
  await browser.close();

 // Return the data
  return {
    url: url,
    title: title.trim(),
    ingredients: [...new Set(ingredients)], // Remove duplicates
    instructions: instructions,
    prepTime: prepTime,
    categories: [...new Set(categories)], // Remove duplicates
    equipment: [...new Set(equipment)] // Remove duplicates
  };
}


async function getRecipeLinks(url, browser) {
    const page = await browser.newPage();
    await page.goto(url);
  
    // Get the links
    const links = await page.$$eval(recipeLinkcss, items => items.map(item => item.href));
  
    await page.close();
  
    return links;
  }

  async function scrapeAllRecipes(baseURL, numberOfPages) {
    let browser;
    try {
      browser = await launch();
    } catch (error) {
      console.error('Failed to launch browser:', error);
      return []; // Exit if browser can't be launched
    }
  
    const allRecipes = [];
  
    console.log(`Starting to scrape recipes from ${numberOfPages} pages...`);
  
    for (let i = 1; i <= numberOfPages; i++) {
      const url = `${baseURL}/page/${i}`;
      let links;
      try {
        links = await getRecipeLinks(url, browser);
      } catch (error) {
        console.error(`Failed to get links from ${url}: ${error}`);
        continue; // Skip to the next iteration if this page fails
      }
  
      for (const [index, link] of links.entries()) {
        try {
          const recipe = await scrapeRecipe(link);
          allRecipes.push(recipe);
          console.log(`Scraped ${index + 1} of ${links.length} recipes from page ${i} (Total: ${allRecipes.length})`);
        } catch (error) {
          console.error(`Failed to scrape ${link}: ${error}`);
        }
  
        // Delay to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  
    try {
      await browser.close();
    } catch (error) {
      console.error('Failed to close browser:', error);
    }
  
    console.log(`Finished scraping ${allRecipes.length} recipes.`);
    return allRecipes;
  }
  
  
  async function runScraper(){

    //Setup mongodb to store the data
    try {
        await mongoose.connect(mongoString, {
        useNewUrlParser: true,
        useUnifiedTopology: true
        });
        console.log('Connected to MongoDB');
    } catch (error) {
        console.error(error);
    };
    //drop the database
    mongoose.connection.db.dropDatabase();

    //define a schema
    const recipeSchema = new mongoose.Schema({
      url: String,
      title: String,
      ingredients: [String],
      instructions: [String],
      prepTime: String,
      categories: [String],
      equipment: [String],
    });
    
    const Recipe = mongoose.model('Recipe', recipeSchema);
 
    // Example usage
    console.log('Scraping recipes...');
    const scrapedRecipes = await scrapeAllRecipes(url, numberOfPages); // Use await here

    //console.log(scrapedRecipes);

    console.log('Saving recipes to MongoDB...');

    for (const recipeData of scrapedRecipes) {
        const recipe = new Recipe(recipeData);
        try {
            await recipe.save(); // Now it's inside an async function
            //console.log('Recipe saved successfully:', recipe.title);
        } catch (err) {
            console.log(err);
        }
    }

    mongoose.connection.close();
};

//start the scraper
runScraper();