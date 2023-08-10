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

const numberOfPages = 1; // Adjust based on how many pages you want to scrape

async function scrapeRecipe(url) {
  // Launch the browser and go to the recipe page  
  const browser = await launch({ headless: "new" });
  const page = await browser.newPage();
  await page.goto(url);

  // Get the data
  const title = await page.$eval('h1', h1 => h1.textContent);
  const ingredients = await page.$$eval(ingredientscss, items => items.map(item => item.textContent));
  const instructions = await page.$eval(instructionscss, div => div.textContent);
  const prepTime = await page.$eval(prepTimecss, div => div.textContent.trim()).catch(() => null); // Replace with correct CSS selector
  const categories = await page.$$eval(categoriescss, items => items.map(item => item.textContent)).catch(() => []); // Replace with correct CSS selector
  const equipment = await page.$$eval(equipmentcss, items => items.map(item => item.textContent)).catch(() => []); // Replace with correct CSS selector


 // Close the browser
  await browser.close();

 // Return the data
  return {
    title: title.trim(),
    ingredients: ingredients,
    instructions: instructions.trim(),
    prepTime: prepTime,
    categories: categories,
    equipment: equipment
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
    const browser = await launch();
    const allRecipes = [];
    let totalRecipesToScrape = 0;
  
    // Loop through all the pages to get the links to each recipe
    for (let i = 1; i <= numberOfPages; i++) {
      const url = `${baseURL}?page=${i}`; // Adjust this URL pattern based on the website structure
      const links = await getRecipeLinks(url, browser);
      totalRecipesToScrape += links.length; // Add the number of links on this page to the total
    }
  
    console.log(`Starting to scrape ${totalRecipesToScrape} recipes...`);
  
    for (let i = 1; i <= numberOfPages; i++) {
      const url = `${baseURL}?page=${i}`;
      const links = await getRecipeLinks(url, browser);
  
      for (const [index, link] of links.entries()) {
        try {
          const recipe = await scrapeRecipe(link);
          allRecipes.push(recipe);
          console.log(`Scraped ${index + 1} of ${links.length} recipes from page ${i} (Total: ${allRecipes.length} of ${totalRecipesToScrape})`);
        } catch (error) {
          console.error(`Failed to scrape ${link}: ${error}`);
        }
  
        // Delay to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  
    await browser.close();
  
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
        title: String,
        ingredients: [String],
        instructions: String,
        prepTime: String, // Optional, you can make this a Number if preferred
        categories: [String], // Array of categories
        equipment: [String], // Array of equipment
      });
    
    const Recipe = mongoose.model('Recipe', recipeSchema);
 
    // Example usage
    console.log('Scraping recipes...');
    const scrapedRecipes = await scrapeAllRecipes(url, numberOfPages); // Use await here

    console.log(scrapedRecipes);

    console.log('Saving recipes to MongoDB...');

    for (const recipeData of scrapedRecipes) {
        const recipe = new Recipe(recipeData);
        try {
            await recipe.save(); // Now it's inside an async function
            console.log('Recipe saved successfully:', recipe.title);
        } catch (err) {
            console.log(err);
        }
    }

    mongoose.connection.close();
};

//start the scraper
runScraper();