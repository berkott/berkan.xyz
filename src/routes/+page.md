<script>
  let profileImage = '/pfp.jpg';
  
  function swapImage() {
    profileImage = profileImage.includes('pfp.jpg') ? '/pfpDrawn.png' : '/pfp.jpg';
  }
</script>

<div class="flex items-center gap-4">
  <div class="relative">
    <button type="button" class="border-0 p-0 bg-transparent" on:click={swapImage} on:keydown={(e) => e.key === 'Enter' && swapImage()}>
      <img src={profileImage} alt="Berkan's profile" class="w-auto h-auto max-w-[180px] max-h-[180px] mt-1 mb-1 object-contain" />
    </button>
    <p class="mt-0 text-xs text-center">(click image to swap)</p>
  </div>
  <div>
    <p>I'm Berkan, AKA Berk. Reach out 😊!</p>
    <p>berkott3002 [at] gmail [dot] com</p>
  </div>
</div>

Professionally, I am a scientist, working to develop a theory of deep learning.

In my private life, I love family, friends, staying healthy, gardening, water polo, the Beatles, playing piano, and more. 🐨✌️❤️.

# Background

## PhD
- PhD student at The University of Pennsylvania. Grateful to be advised by [Surbhi Goel](https://www.surbhigoel.com/).
- Interning with [James Simon](https://james-simon.github.io/) at [Imbue](https://imbue.com/).
- Data scientist intern at [Hawaii Farming](https://www.hawaiifarming.com/).

## Undergrad
- Completed my undergrad at Columbia (CS major, applied math minor). Grateful to be advised by [Daniel Hsu](https://www.cs.columbia.edu/~djhsu/).
- Garden manager at the [Columbia Gardening Club](https://gardening.studentgroups.columbia.edu/).
- Summer research assistant at The Center for Computational Math at the Flatiron Institute. Grateful to be mainly advised by [Berfin Şimşek](https://www.bsimsek.com/) and helped by [Denny Wu](https://dennywu1.github.io/).
- Head teaching assistant for Analysis of Algorithms (2 times) at Columbia.
- Teaching assistant for Machine Learning (4 times) at Columbia.

## High school
- President and programming head of the [Techno Titans](https://technotitans.org/), FRC team 1683.
- Software engineering intern at [State Farm](https://statefarm.com/).
- Machine learning / full stack intern at [NimbleThis](https://nimblethis.com/) and [Senslytics](https://senslytics.com/) in Atlanta.
